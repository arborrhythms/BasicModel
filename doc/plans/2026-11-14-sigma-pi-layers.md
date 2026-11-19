# SigmaLayer and PiLayer: a sum, a product, and no chart that only bounds (Claude, 2026-11-14)

A specification that began as a rewriting of `SigmaLayer` and `PiLayer`
of `bin/Layers.py`; by §18.24, rule 32, those two classes are left as
they are and new classes are built. It was round D of item 1.8 (§18). Since 2026-11-16 it
is item 1.7 (Alec: "I think we should position D as item 1.7"), built
after item 1.8 is finished and pushed; it keeps the name round D here.
**Its rules to build, with the text for Codex, are §19, which is the
last word of this specification;** everything before §19 is the record.
A rule marked *default* is Claude's choice where Alec has not spoken.

**Where each section stands.** §18.5 to §18.8 are the rules as
decided, their certificates and Claude's defaults. The rest is the
record:

- §1 to §3: Alec's first words, what is built, and the first toy.
  They stand.
- §4 to §7: the first rules. Superseded by §18.5: Sigma keeps a `tanh`
  at its exit and is not clipped, and the inventory of every other
  chart in `bin/` (§4, rule 7) is not in round D.
- §8 to §10: the order of the stack, sigma over pi in the
  architecture, and the paired product of March. A record.
- §11 to §17: the paired Pi, explored and measured. Not built:
  `PiLayer`, the unpaired one, is kept (§18.1). §16's identities are
  used by §18.18.
- §18.3 and §18.4: two tests of regularization. Not built.
- §18.9 to §18.21: the questions and Alec's answers that led to
  §18.5.

## 1. Alec's words (2026-11-14)

"If SigmaLayer and PiLayer are reducing to a linear map, then we need to
remove those tanh. In principle, we need no tanh except as used in
PiLayer to take us into log domain so that we can compute a matrix
exponential with the weight matrix instead of a matrix product, as in
sigma layer. The original design is probably clear from the older
experiments in bin/etc. Sigma is basically just a weight matrix, and I
don't even think it needs a tanh when combined with Pi for nonlinearity.
Pi was originally a simple cumulative product instead of what it is now,
but basically achieved this by being a matrix product in the log domain.
So if you can make a spec for rewriting those to be efficient and to
remove as many tanh and atanh as possible, while ensuring that inputs and
outputs are bounded 0..1 or -1..1, please do."

Earlier, in FutureWork: "it seems that the tanh and atanh were added as
excessive precautions against unit overflow: we can just use clipping and
some remedy for saturation".

And, the same day: "when I use those terms, I am almost always talking
about the SigmaLayer and PiLayer as we have designed them."

## 2. What is built, and what was built first

Read at `3926005f12`.

- **`SigmaLayer`** with `nonlinear=True` is `tanh(atanh(x) W + b)`: a
  chart on the way in, the weight matrix, the chart's inverse on the way
  out (`_chart`, `_unchart`; `cube_chart` and `cube_unchart` where
  `unit_interval` is set).
- **`PiLayer`** with `nonlinear=True` is `tanh(atanh(x) W + b/2)`. Its own
  account is the product: the entry is `log((1+x)/(1-x))`, which is
  `2 atanh(x)`, the weight matrix acts in that log domain, and the exit
  `(y-1)/(y+1)` of the exponential is `tanh` of half the result.
- **Stacked, they are one linear map inside one chart.** Sigma's entry
  `atanh` cancels Pi's exit `tanh`, and the reverse order cancels the
  same way. `SigmaLayer`'s own description says it: stacking "does not
  create a hidden nonlinear feature layer".
- **Where they are built in production:** `ChangeLayer._sigma`
  (`bin/Language.py`, where there is no meaning layout), the additive
  readout that already passes `nonlinear=False` (`bin/Layers.py`, its
  comment: "NOT its legacy atanh input chart"), and the `pi` and `sigma`
  natural folds a grammar file may name (`Language.py`, the
  `{'pi': PiLayer, 'sigma': SigmaLayer}` dispatch). `SigmaLayer2` and
  `PiLayer2` are kernel forms of the same two.
- **The first design** (`bin/Model.py` at `17bee20529`, 2026-03-11).
  Sigma was the weight matrix with one `tanh` on its output and no chart
  on its input. Pi was a product over its inputs of `1 + tanh(w x + b)`,
  taken with `torch.prod`. Its comment records why the log domain was not
  used then: "Tried to implement as exp( SIGMA(log(e) * log(wx)) ) so that
  we could invert with standard methods". `bin/etc/SigmaPi.py` trains a
  Pi layer of three units under a Sigma layer on XOR.
- **Counted:** 128 calls of `tanh`, `atanh`, `cube_chart`, `cube_unchart`
  or `bounded_atanh` in `bin/`, 60 of them in `Layers.py`, 39 in
  `Language.py`, 8 in `MeaningKernels.py`.

## 3. Checked as a toy

Arithmetic in the session's scratch directory, not the model and not a
receipt: XOR by a Pi layer of three units under a Sigma layer, twenty
random starts each, 3,000 steps.

| Form | XOR solved |
|---|---:|
| As built: both layers charted | 0 of 20 |
| Pi as built; Sigma a weight matrix, clipped (§4, rules 1 and 2) | 20 of 20 |
| No `tanh` at all: Pi as `exp(log(x) W)` on the unit range, `W >= 0` | 4 of 20 |
| The same, with each input also given as `1 - x` | 15 of 20 |

So the as-built stack cannot learn it, removing Sigma's chart alone is
enough, and the product with no chart at all is expressible but trains
badly: a product over an input at zero is dead from the start.

## 4. The rules

1. **Sigma is a weight matrix.** `y = x W + b`, with no chart on the way
   in or out. Its output is held in range by clipping, to 0..1 where
   `unit_interval` is set and to -1..1 otherwise. *Default* for the
   "remedy for saturation": the value is clipped and the gradient passes
   as if it were not, and the share of outputs clipped is reported.
2. **Pi is a product in the log domain, and keeps the one chart that
   takes it there.** For the range -1..1 it is `tanh(atanh(x) W + b/2)`,
   which is the product of the inputs' odds raised to the weights. For
   the range 0..1 it is the same in the cube's chart: the entry is the
   log-odds `log(x / (1 - x))` and the exit is its inverse. One entry and
   one exit for a call; these are the logarithm and the exponential of
   the product, and not a bound added to it.
3. **Stacked, Pi under Sigma is nonlinear,** because nothing cancels Pi's
   exit. Sigma under Pi is too.
4. **One form each.** `SigmaLayer`'s `nonlinear` flag goes, since both of
   its values are now the weight matrix. `PiLayer`'s `nonlinear=False`,
   which returns the unbounded exponential, goes (*default*; its users
   are listed and each is shown to keep its range). `PiLayer` takes
   `unit_interval` as `SigmaLayer` does.
5. **Reverse.** Sigma's reverse is the inverse matrix on `y - b`, clipped
   the same way; it is exact where nothing was clipped, and the layer
   says where it was. Pi's reverse is unchanged.
6. **Efficient.** A call is one matrix product; Pi adds one `atanh` and
   one `tanh` (or the log-odds and its inverse) and nothing else. The
   detours through `_to_mult`, `log`, `exp` and `_from_mult` go where the
   closed form is the same number. The bounded slope of the entry chart
   (`_BoundedLogMult`, `bounded_atanh`) stays, since Pi still has an
   entry. The butterfly's per-pair operations follow rules 1 and 2.
   `SigmaLayer2` and `PiLayer2` follow them too.
7. **Every other chart is sorted by what is inside it.** Each of the 128
   calls is listed and marked: the entry or exit of a product (it stays:
   `PiLayer`, and the gains of `MeaningAction` and `LaneKernel`, which
   multiply in the chart); a chart around a sum or an affine map that
   only keeps it in range (it goes, and the result is clipped); or
   neither, which is reported and left. A call whose kind is not plain is
   a question for Alec and is not guessed.
8. **Bounds are asserted, not assumed.** Inputs and outputs are in 0..1
   or -1..1 at every layer boundary, the rails included, and nothing
   non-finite is clipped into range: it still fails loudly.

## 5. What it changes

- Values change wherever a `SigmaLayer` is in the path: the grammar's
  `lift` where there is no meaning layout, and any `sigma` fold a grammar
  file names. Nothing is expected to stay equal there.
- `PiLayer`'s values do not change for `nonlinear=True`.
- The stacks of item 1.8's round B (`expect`, `intuit`, the kernels of
  `marker`) are a `PiLayer` under a `SigmaLayer` built by these rules,
  in place of the product and probabilistic sum that §30.2 of the 1.8
  specification wrote. That section was Claude's design and was not what
  Alec meant by a sigma-pi.

## 6. Certificates

1. *Not linear.* On random inputs, a `PiLayer` under a `SigmaLayer` is
   not an affine function of the charted input, and the as-built pair is
   (the before case).
2. *XOR.* `bin/etc/SigmaPi.py`'s network, three Pi units under one Sigma
   unit, learns XOR; the count over a stated number of ordinary starts is
   measured first and the bar set from it. No start is chosen.
3. *Bounds.* For inputs over the whole range, the rails included, and
   weights scaled far beyond their start, every output of both layers is
   finite and in range, for both ranges.
4. *The charts that remain.* A test reads the source: `SigmaLayer` calls
   no `tanh` or `atanh`; a `PiLayer` call makes one entry and one exit.
5. *Reverse.* Pi's reverse returns its input (atol 1e-6). Sigma's does
   where nothing was clipped, and reports the rest.
6. *The inventory.* The 128 calls with the mark of rule 7 for each, and
   the count that remains.
7. *Gates and seconds.* Both stopping gates hold. Seconds for a forward
   and a backward call of each layer, before and after.

## 7. Not decided

- Who builds it and when. `bin/Layers.py` is being rewritten by item
  1.8's rounds A and B, so it cannot be built beside them without a
  conflict; and round B's stacks should be built on it.
- Whether the clipped gradient passes straight through (rule 1's
  default), or the weights are kept so that Sigma cannot leave its range.
- Whether `PiLayer`'s unbounded form has a user that needs it (rule 4).

## 8. The order of the stack: a dialog (2026-11-14)

Alec asked for Pi written as a product, and for the stack he had in mind
for `intuit` with the Sigma expression inlined in the Pi.

**Pi as a product.** Write the odds of a value as
$o(x) = (1+x)/(1-x)$. Since $\operatorname{atanh}(x) = \tfrac12 \log
o(x)$, the Pi of §4 is

$$P_j = e^{b_j} \prod_i \left(\frac{1+x_i}{1-x_i}\right)^{W_{ij}},
\qquad y_j = \frac{P_j - 1}{P_j + 1}.$$

In the log domain $\log P_j = b_j + \sum_i W_{ij} \log o(x_i)$, a matrix
product. For the range 0..1 the odds are $x/(1-x)$ and the way back is
$P/(1+P)$.

**Sigma inlined in Pi** (Sigma underneath):

$$P_j = e^{b_j} \prod_k \left(\frac{1 + c_k + \sum_i V_{ik} x_i}
{1 - c_k - \sum_i V_{ik} x_i}\right)^{W_{kj}},
\qquad y_j = \frac{P_j - 1}{P_j + 1}.$$

**The other order** (Pi underneath):
$z_m = d_m + \sum_j U_{jm}\,(P_j - 1)/(P_j + 1)$ with $P_j$ the first
product above.

**The toy** (XOR, three hidden units, twenty starts; arithmetic, not the
model): Pi underneath solves it 20 of 20, Sigma underneath 1 of 20.

- Alec: "Why the strong nonlinearity?" Claude: a product runs from zero
  to infinity, and bringing it back into range by $(P-1)/(P+1)$ makes an
  S-curve, a soft threshold; a sum of such thresholds placed apart makes
  a bump, which is XOR. With Sigma underneath, the only curve is the
  logarithm of a sum's odds, which is nearly straight until the sum is
  near a rail and there grows without limit.
- Alec: "Why do you prefer it?" Claude: Pi underneath is the order of
  Alec's own account ("pi(percepts) -> concepts", concepts as sigma of
  pi); a hidden Pi unit can be read as a conjunction of inputs; and it
  trained in the toy. The toy is one small problem and not the model.
- Alec: "But I like a big linear region because most gradient methods
  work well there." Claude: both orders are linear through the middle
  and differ at the extremes, where one flattens and the other steepens;
  a stack that stays in its linear region computes a linear map, which
  cannot give an expectation that depends on a combination. A direct
  linear path beside the product path,
  $z = \Sigma_{\text{direct}}(x) + \Sigma_{\text{out}}(\Pi(x))$, would
  keep a linear region everywhere and leave the products the
  corrections. That form is not tested.

Not decided: which order `intuit`'s stack takes.

## 9. Sigma over Pi in the architecture (verified, 2026-11-14)

Alec: "I think we are actually doing sigma over pi in the architecture,
as concepts over percepts. Please verify that." It is so, in two places,
read at `3926005f12`:

- *A basic percept is a least.* Its presence is the least of its wholes'
  properties (`BasicPercepts.property_presence`, `amin`), and its code
  the least of its parts' codes. That is the Pi.
- *An order-zero concept is a Sigma over basic percepts.*
  `IndexedSigmaConceptsFromPercepts` ("Plain Sigma readouts") holds one
  affine row for a concept over its references to percepts.
- *A defined concept is a Sigma of a Pi over concepts.*
  `ConceptualSpace._compose_order0` first folds a row's conjunctive
  edges by the least (`store.conjunctive.fold_presence(...,
  conjunctive=True)`) where `conceptualPi` is on, and then folds the
  result with the row's other edges by the greatest
  (`store.fold_presence`).

So the order is Sigma over Pi. None of these three uses the classes
`SigmaLayer` or `PiLayer`: the Pi is a least (of powers, in the fold),
and the Sigma is a greatest in the fold and an affine readout over
percepts. The two classes are used by the grammar's `lift` and by the
`pi` and `sigma` folds a grammar file may name.

## 10. The paired product (the form before the matrix in the log domain)

Alec: "I'd like also to explore the form we had prior to the matrix
exponential, which required twice as many outputs in order to do the
inversion. (1+x and 1-x pairs). That was my original simple understanding
of how the cumprod would work, and it still feels more natural than the
matrix exponential even though it doesn't inver with as few terms."

**What it was** (`InvertiblePiLayer`, `bin/Model.py` at `fb96a7e968`,
2026-03-16). For each unit $j$, with $t_{ji} = \tanh(w_{ji} x_i)$:

$$z_j = \prod_i (1 + t_{ji}), \qquad y_j = \prod_i (1 - t_{ji}).$$

Its reverse used both: $\tfrac12(\log z_j - \log y_j) = \sum_i w_{ji}
x_i$, so the pair gives $W x$ exactly and $x$ follows by the inverse of
$W$. That is why it had twice the outputs.

**What it is, read today.**

- With each factor halved, $(1 \pm t)/2$, both products lie in 0..1 with
  no clipping. $z$ is high when every input pushes the unit one way and
  $y$ when every input pushes it the other; both are low when the inputs
  disagree. The pair is the two evidence lanes, for and against, as two
  positive numbers that are never netted.
- The single output of §4 is the pair netted:
  $(z - y)/(z + y) = \tanh(\sum_i w_{ji} x_i)$. Netting discards what
  the pair keeps, how far the inputs agreed at all.
- The weight sits inside each factor, so a unit is not one matrix
  product: it is a `tanh` (or a log-sigmoid) for each pair of input and
  unit, then a sum of logarithms. That is the cost in speed, of the same
  order as a matrix product but not one.

**The toy** (XOR under a Sigma that is a clipped weight matrix, twenty
starts):

| Inputs | One pair | Three pairs |
|---|---:|---:|
| -1..1 | 17 of 20 | 20 of 20 |
| 0..1 | 3 of 20 | 20 of 20 |

One pair is enough in principle, since $z + y$ is "the inputs agree" and
XOR is its complement. Every pair stayed in 0..1.

**For Alec.** Whether Pi becomes the paired product. It would give each
unit two outputs, which the two lanes of a concept already are; it needs
no clip and no chart beyond the `tanh` inside each factor; it loses the
single matrix product; and its inverse needs the pair.

## 11. Decided (Alec, 2026-11-14)

"Let's creat both pairedPi and Pi, with the latter being default. Intuit
should be sigma(pi(x))."

- There are two layers, `PiLayer` and a paired Pi (*default* name
  `PairedPiLayer`). `PiLayer` is the default wherever a Pi is asked for.
- `intuit`'s stack is a Pi underneath and a Sigma on top. This settles
  §8's open point.

## 12. The paired Pi, explored (for Alec; nothing here is decided)

Alec: "I want to explore the pairedPi more closely, I think it offers the
possibility of regularizing over the output such that it becomes easy to
create a subspace. I'd also like to see how it relates to tetralemma
logic, since we have and/or in Boolean sigma and pi, and four corners
over the extremal values of the pairedPi outputs."

### 12.1 Two ways to write it

- *The March form* (§10): the weight inside each factor,
  $z_j = \prod_i (1 + \tanh(w_{ji} x_i))/2$ and its mirror. An input the
  unit should ignore ($w = 0$) still halves both outputs, so a unit over
  many inputs can never come near one.
- *Weights as powers* (Claude's proposal). Give each input as its two
  lanes, $a_i$ for and $b_i$ against (for a one-lane value in -1..1,
  $a = (1+x)/2$ and $b = (1-x)/2$: the "1+x and 1-x pairs"). Then

  $$z_j = \prod_i a_i^{G_{ji}}, \qquad y_j = \prod_i b_i^{G_{ji}},
  \qquad G \ge 0.$$

  In the log domain each lane is one matrix product,
  $\log z = G \log a$ and $\log y = G \log b$, so it is the matrix form
  with two outputs. An input with weight zero drops out. It has no
  `tanh` and no `atanh` at all, and both outputs are in 0..1 by
  construction. A weight of the other sign is the same weight on the
  other lane of that input.

Checked as arithmetic (the session's scratch directory; not the model):

- *Netting the pair gives today's `PiLayer`:*
  $(z - y)/(z + y) = \tanh(\sum_i G_i \operatorname{atanh} x_i)$, to six
  places. So `PiLayer` is the paired Pi with its two lanes netted into
  one number, and the paired Pi is `PiLayer` before that is done.
- *The inverse needs the pair:*
  $\tfrac12(\log z - \log y) = \sum_i G_i \operatorname{atanh} x_i$.

### 12.2 The four corners

With a unit's weights summing to one, so that a unit is a weighted
geometric mean of each lane, and two inputs at one half each:

| Inputs | for $z$ | against $y$ | $z + y$ | Corner |
|---|---:|---:|---:|---|
| both for | 1 | 0 | 1 | true |
| both against | 0 | 1 | 1 | false |
| one for, one against | 0 | 0 | 0 | neither |
| both unknown, lanes (0, 0) | 0 | 0 | 0 | neither |
| both "both", lanes (1, 1) | 1 | 1 | 2 | both |
| both undecided, lanes (.5, .5) | .5 | .5 | 1 | midway from true to false |
| $x = .9$ and $x = -.3$ | .577 | .180 | .757 | pulled toward neither |

- *One-lane inputs stay on or under the line from true to false.* Over
  20,000 random cases $z + y$ never exceeded one, and it equals one
  exactly when the inputs agree. What is missing from one,
  $1 - z - y$, measures how far the inputs disagree. A paired unit
  therefore reports three things: for, against, and how mixed its inputs
  were.
- *The corner "both" is reached only from inputs that are themselves
  both.* Inputs that disagree with each other read as neither.
- *The mirror unit reads disagreement as both.* Taking the product over
  the complements, $z' = 1 - \prod (1-a_i)^{G}$ and
  $y' = 1 - \prod (1-b_i)^{G}$, one input for and one against gives
  $(1, 1)$.
- *How this sits beside and and or.* The field's `intersection` and
  `union` (§3.2 of the 1.8 specification) order the corners by truth:
  and takes the least of for and the greatest of against. The paired Pi
  takes the product on both lanes, which is the other order of the same
  four corners, by how much is known: it keeps what every input agrees
  on (in Belnap's four-valued logic, consensus), and its mirror keeps
  what any input says (gullibility). And and or are each one lane of
  the paired Pi with the other lane of its mirror: and is
  $(\prod a,\ 1 - \prod (1-b))$.
- *The same three readings drive attention today.*
  `Attention.narrowing_mask` takes pure, both and neither of a bracket:
  pure is what $z + y$ near one says of a paired unit.

### 12.3 Regularizing the output, and a subspace

Three things the pair makes possible that one netted number does not.
Which of them Alec means is for him to say.

1. *Off is a corner.* A netted unit's zero lies midway between its two
   extremes, so a penalty that pushes units toward zero also pushes
   them toward "undecided". A paired unit's off state is $(0, 0)$,
   apart from both of its decided states. A penalty on $z + y$ leaves
   most units off and a few decided, each signed by its lane. The units
   that are on are a subspace: the reverse of §12.1 recovers the input's
   part in the span of their weight rows and nothing else.
2. *The line from true to false is a constraint that can be asked for.*
   A penalty on $1 - z - y$ asks each unit to see inputs that agree,
   which is to ask each unit's inputs to be of one kind.
3. *Two numbers for a unit let two penalties act apart:* one on how
   often a unit is on, one on how decided it is when it is on.

### 12.4 Questions

- Which form of the paired Pi: the March form, or weights as powers
  (§12.1)? Claude prefers the second: it is the matrix form, it has no
  chart, and `PiLayer` is exactly its netted case.
- Which regularizer, and what the subspace is for.
- Whether a paired Pi's weights are kept summing to one for a unit.

## 13. The paired Pi, as Alec set it (2026-11-14)

Alec:

1. "Weights as powers is great. Please confirm that combining this with
   sigma will give us nonlinear learning with no transcendentals. If
   it's bounded everywhere, even better (I.e. do we need to clip to keep
   unit range)"
2. "The point is to reduce the number of terms, so that a sparse weight
   matrix can do PI (many terms has numerical issues)"
3. "Weights can vary, but the signal in/out needs to be guaranteed at
   0..1 for percepts and -1..1 for concepts, because they live in
   positive unit hypercube and unit hypersphere."

**What can be confirmed, and one correction.**

- *A power with a real exponent is itself a transcendental:* $a^{G}$ is
  computed as $\exp(G \log a)$. So weights as real powers have no `tanh`
  and no `atanh`, but they do have a logarithm and an exponential.
- *With no transcendental at all,* the weight enters each factor as a
  gate between leaving the input out and taking it whole:

  $$z_j = \prod_i \bigl(1 - g_{ji}\,(1 - a_i)\bigr), \qquad
  y_j = \prod_i \bigl(1 - g_{ji}\,(1 - b_i)\bigr), \qquad 0 \le g \le 1.$$

  At $g = 0$ the input drops out, at $g = 1$ the factor is the input,
  and these two are the powers zero and one. Only additions and
  multiplications are used. The pair netted equals `PiLayer` where every
  gate is zero or one.
- *Nonlinear, and it learns.* A Sigma over such pairs is a sum of
  products, a polynomial in the inputs. In a toy (XOR, three paired
  units under one Sigma unit, twenty starts, arithmetic only and not the
  model) it was solved 20 of 20 with a margin above one half.
- *Bounded everywhere, with no clip of a signal.* A product of numbers
  in 0..1 is in 0..1, so the paired Pi needs only its gates kept in
  0..1. A Sigma stays in range for every possible input when its weights
  do: for an output in -1..1 over hidden values in 0..1, the bias plus
  the positive weights is at most one and the bias minus the negative
  weights is at least minus one; for an output in 0..1 the same with the
  lower bound at zero. The weights are returned to that set after each
  step (a projection, as `SparseLayer` already keeps its magnitudes).
  In the toy the output never left -1..1 over 5,000 random inputs for
  each start, and no signal was clipped.
- *Sparse.* Each unit has few terms: the gates are a sparse matrix, an
  absent entry is a gate of zero, and the product runs over the entries
  present. That is the storage the definition store's conjunctive edges
  and `IndexedSigmaConceptsFromPercepts`' references already have.

**The rules, replacing §12.4's first and third questions.**

1. `PairedPiLayer` takes each input as two lanes in 0..1 and gives two
   lanes in 0..1 for each unit, by the gated product above over a sparse
   matrix of gates in 0..1. It uses no transcendental function.
2. `SigmaLayer` (§4, rule 1) holds its range by its weights, and no
   signal is clipped (this replaces §4's clip and its *default*). Its
   range is 0..1 where it gives percepts and -1..1 where it gives
   concepts.
3. `PiLayer` stays as §4 has it and stays the default (§11).
4. *Default:* a unit's number of terms is bounded by its storage, as a
   concept's references to percepts are today.

**One point for Alec.** These rules bound each coordinate. A concept on
the unit hypersphere also has length one, which a bound on coordinates
does not give; that needs the vector divided by its length, as the
codebook does now, and it is not a transcendental (a square root).
Whether a Sigma that gives concepts ends with that division is his to
say.

## 14. From the pair to a concept, and what it costs (2026-11-14)

Alec: "It is in fact a hyperball, but the magnitude is reserved for
certainty. But that certainty and the sign are a kleene 3 value logic,
and we have a 4 valued logic from the pairedPi, so the pi values should
be summed or netted somehow after their use in regularization to get the
appropriate range for concepts."

**The rule.** A paired unit's two lanes are used as two for
regularization (§12.3). They are then netted to one signed number,
$c_j = z_j - y_j$, in -1..1: its sign says which lane, and its size is
the certainty. True is $+1$, false $-1$, and both "neither" $(0, 0)$ and
"both" $(1, 1)$ net to zero, which is the unknown of the three-valued
logic. The Sigma above takes the netted values. Nothing is divided by a
length, since the length is the certainty.

*Default:* the difference, and not the ratio $(z - y)/(z + y)$ that
`PiLayer` amounts to. The ratio is at full size for the weakest evidence
and is undefined at $(0, 0)$, so it does not carry certainty.

*For Alec:* whether the bound of one is on each coordinate, which these
rules give, or on the length of the whole vector, which a bound on
coordinates does not give.

**Whether it is faster than a rectified linear unit.** Alec: "Would you
expect this version, with no transcendentals, to be any faster than the
modern rectified linear unit transfer functions?" No. One hidden layer
of 1,024 units over 1,024 inputs at a batch of 256, forward and
backward, as a micro-benchmark on this machine (not the model):

| Layer | Processor | MPS |
|---|---:|---:|
| Rectified linear, dense | 0.61 ms | 0.38 ms |
| Rectified linear over 8 inputs a unit | 0.89 ms | 0.68 ms |
| Gated paired product over 8 terms a unit | 3.13 ms | 1.47 ms |
| Today's `PiLayer` form, dense | 0.94 ms | 0.25 ms |

The paired product is four to five times slower for a layer than the
rectified linear unit. The cost is the product over a unit's terms and
its backward pass, and the gathering of sparse terms; a dense matrix
product is the most optimized operation there is. The transcendentals
are not what costs: today's `PiLayer` form is as fast as the rectified
linear layer. So removing them buys exact bounds, an exact zero and no
clipping, and not speed. What the pair may save is units, since one
paired unit holds a conjunction that takes several rectified units; that
is not measured.

**A correction to the table above (Alec: "Did your gated paired product
use a dense matmul?").** It did not. The gated form gathers eight inputs
for each unit and multiplies them, with no matrix product; a product of
gated factors cannot be written as one. The paired Pi with weights as
powers can: with the logarithm of each lane taken once for the input,
each lane is `exp(log(a) @ G)`. Measured the same way:

| Layer | Processor | MPS |
|---|---:|---:|
| Rectified linear, dense | 0.59 ms | 0.37 ms |
| Paired Pi, weights as powers, dense matrix product in the log domain | 1.30 ms | 0.40 ms |
| Paired Pi, gated, eight gathered terms a unit | 3.11 ms | 1.48 ms |

So the two forms trade against each other: the gated form has no
transcendental and no matrix product, and is the slow one; the power
form has one logarithm for the input and one exponential for the output
and runs at the speed of a rectified linear layer on an accelerator.

**Trying to make the gated form faster (Alec: "I think there must be a
way to compute the gated version more efficiently.").** Five ways were
measured at the same size (1,024 units of eight terms, batch 256,
forward and backward; milliseconds):

| Way | Processor | MPS |
|---|---:|---:|
| Rectified linear, dense (for comparison) | 0.62 | 0.42 |
| Gather, then `prod` | 3.18 | 1.69 |
| Gather, then three explicit pairwise products | 3.87 | 1.46 |
| Eight neighbouring inputs as a view, no gather | 3.18 | 1.59 |
| Fewer passes (`index_select`, `addcmul`), compiled | 2.66 | not run |
| At most two terms a unit, exact, by matrix products only | 3.14 | 1.07 |

The last uses $\prod (1 - u_i) = 1 - \sum u_i + \tfrac12[(\sum u_i)^2 -
\sum u_i^2]$ with $u_i = g_i (1 - a_i)$, which is exact for two terms
and needs two matrix products for a lane. None of the five comes within
a factor of two and a half of the rectified linear layer. The gather is
not the cost and neither is `prod`: the work is several passes over a
tensor of batch by units by terms, where a matrix product is one tuned
kernel. Claude found no formulation of the gated product as a single
matrix product.

Two things put this in proportion. The cost is for each term stored, so
128 units of eight terms cost an eighth of the figures above. And a
lesson sentence takes about four seconds today, nearly all of it host
work (§10 of the performance branch's plan), so a layer of one to three
milliseconds is not what limits the model.

**The whole stack (Alec: "is your pairedPi calculation also doing the
sigma?").** It was not: the figures above are the Pi layer alone, netted,
against a rectified linear layer that includes its own matrix product.
With the Sigma on top of each, 1,024 inputs to 1,024 hidden to 1,024
outputs (milliseconds):

| Stack | Processor | MPS |
|---|---:|---:|
| Rectified linear hidden layer, linear output | 1.59 | 0.89 |
| Gated paired Pi, netted, Sigma | 4.39 | 2.09 |
| Paired Pi with weights as powers, netted, Sigma | 2.45 | 0.75 |

## 15. The target, and why the pair is not netted before Sigma (2026-11-14)

Alec: "So the pairedPi with weights as powers, netted, then sigma, is the
best on a number of counts. Let's target that as our theoretically
justified version of the MLP layer." And then: "Oh. Also, I need
inversion, which may be another reason netting will not work."

**The target.** The paired Pi with weights as powers under a Sigma is the
model's form of a layer with one hidden layer. It replaces §13's gated
form as the paired Pi's arithmetic: one logarithm of the input lanes,
one matrix product, one exponential.

$$z_j = \prod_i a_i^{G^{+}_{ij}}\, b_i^{G^{-}_{ij}}, \qquad
y_j = \prod_i b_i^{G^{+}_{ij}}\, a_i^{G^{-}_{ij}}, \qquad G^{+}, G^{-} \ge 0.$$

$G^{-}$ is a weight of the other sign: it takes the other lane of that
input.

**Sigma reads both lanes; nothing is netted before it.** Two reasons.

1. *A netted stack cannot learn most functions.* Complementing every
   input exchanges $z$ and $y$, so $z - y$ changes sign. A Sigma over
   netted units can therefore give only functions $f$ with
   $f(x) + f(\bar x)$ constant. XOR is not one. In the toy below the
   netted stack fit XOR from 0 of 20 starts at any width, and a random
   truth table from 0 of 20; it fit parity of three inputs, which
   happens to have that symmetry.
2. *The inverse needs the pair* (Alec). With $W = G^{+} - G^{-}$,
   $\log z - \log y = W\,(\log a - \log b)$, so the input is recovered
   from the pair by the inverse of $W$ and the inverse of the log-odds;
   checked to $10^{-7}$. From $z - y$ alone it cannot be.

So the two lanes go to Sigma as two inputs with weights of their own,
and Sigma's signed weights do the netting: its output, in -1..1, is the
three-valued quantity of §14. The pair itself is what is regularized
and what is inverted. This replaces §14's "the Sigma above takes the
netted values".

**Against a rectified linear stack** (toy, 3,000 steps, twenty starts
each; "fit" is an error under one thousandth; not the model):

| Task | Hidden | Rectified linear | Paired Pi, as many pairs | Paired Pi, same weights |
|---|---:|---:|---:|---:|
| XOR | 4 | 13 of 20 | 20 of 20 | 20 of 20 |
| Parity of three | 4 | 3 of 20 | 12 of 20 | 0 of 20 |
| Parity of three | 16 | 20 of 20 | 20 of 20 | 20 of 20 |
| A random truth table of four inputs | 4 | 2 of 20 | 0 of 20 | 0 of 20 |
| A random truth table of four inputs | 16 | 20 of 20 | 20 of 20 | 12 of 20 |
| $x_1 x_2 + (1 - x_3)/2$ on continuous inputs | 4 | 17 of 20 | 20 of 20 | 20 of 20 |

"As many pairs" gives the paired stack twice the weights of the
rectified one (two lanes in, two lanes out); "same weights" gives it
half as many pairs. On the continuous target the paired stack's error
is zero to five places, since the target is itself a sum of products.

Seconds for the full form, both lanes crossed and both read by Sigma
(1,024 inputs and outputs, batch 256, forward and backward,
milliseconds):

| Stack | Processor | MPS |
|---|---:|---:|
| Rectified linear, 1,024 hidden | 1.51 | 0.77 |
| Paired Pi, 512 pairs (the same weights) | 2.60 | 0.78 |
| Paired Pi, 1,024 pairs (twice the weights) | 4.78 | 1.23 |

**In sum.** At the same number of weights the paired stack costs the
same as the rectified one on an accelerator and about 1.7 times on the
processor. It learns products and agreements with fewer units and
exactly; it is weaker on an arbitrary table at the same weights; it is
bounded without a clip; and it has an exact inverse, which the rectified
unit does not.

## 16. Netting and the inverse; the pair from the unpaired layer (2026-11-14)

Alec: "So we have a nice theoretical pairedPi that produces 2x the
outputs. What are the reconstruction errors if we eliminate the both
term, and net the result? Is there a way to produce those pairs for the
unpaired version?"

**The pair is the unpaired value and one more number.** Write the pair
as its sum and its difference, $s = z + y$ and $d = z - y$, and let
$c$ be today's `PiLayer` value, $c = \tanh(W\,\operatorname{atanh} x)$
with $W = G^{+} - G^{-}$. Then, exactly,

$$c = \frac{z - y}{z + y}, \qquad d = c\,s, \qquad
z = \frac{s\,(1 + c)}{2}, \qquad y = \frac{s\,(1 - c)}{2}.$$

So the unpaired layer is the pair with $s$ divided out, and the
difference is the unpaired value times $s$. The part common to the two
lanes, $\min(z, y)$, cancels in $z - y$: taking it out and then netting
is the difference $d$.

**Reconstruction** (arithmetic, not the model: eight inputs in -1..1
with a mean size of 0.4, a square $W$ whose columns sum to one in
absolute value, 2,000 samples; the mean absolute error of the
recovered input):

| What is kept for each unit | Error |
|---|---:|
| The pair $(z, y)$ | 0 |
| The ratio $c$ alone (the unpaired layer) | 0 |
| The difference $d$ and the sum $s$ | 0 |
| The difference $d$ alone | 0.04 to 0.10 |
| The two lanes, each less $\min(z, y)$ | 0.7 to 0.9 |

The zeros are $10^{-7}$. The difference alone loses a tenth to a quarter
of the input's size, because $d = c\,s$ and nothing says how much of a
small $d$ is a small $c$ and how much a small $s$; in the trial $s$ ran
from 0.34 to 1.00 with a mean of 0.90. Taking the common part out of
each lane and keeping the lanes leaves one lane at zero, and the
logarithm the inverse takes of it is unbounded.

**So netting does not cost the inverse; netting to one number that also
carries $s$ does.** The inverse needs $c$ and nothing else. What the
second number of a pair adds is $s$, which the inverse does not use.

**The pair, produced for the unpaired layer.** Two ways.

1. *From $c$ alone:* $\big((1 + c)/2,\ (1 - c)/2\big)$, the same map
   that makes the lanes of an input. This is the pair with $s$ set to
   one. It has the same inverse and holds no "neither" and no "both".
2. *The true pair, with one more product.* With
   $\ell_a = \log a$, $\ell_b = \log b$:

$$r = (\ell_a - \ell_b)\,W, \qquad m = (\ell_a + \ell_b)\,|W|, \qquad
z = e^{(m + r)/2}, \qquad y = e^{(m - r)/2}.$$

   $r$ is twice what `PiLayer` computes today ($c = \tanh(r/2)$), and
   $m$ is one more matrix product, of $\log(a\,b) = \log\frac{1 - x^2}{4}$
   with the absolute weights. It equals the four-product form of §15 to
   $10^{-13}$. So `PairedPiLayer` is `PiLayer` with a second output,
   and the two share one weight matrix.

One layer, 1,024 inputs and outputs, batch 256, forward and backward,
milliseconds:

| Layer | Processor | MPS |
|---|---:|---:|
| Rectified linear | 0.66 | 0.40 |
| Unpaired Pi (today's chart) | 0.95 | 0.41 |
| The pair by four products (§15) | 3.61 | 1.08 |
| The pair by two products | 2.49 | 0.63 |

**What $s$ is** (Claude's reading). $s$ is $1$ where a unit's inputs,
each taken on the lane its weight's sign selects, agree, and falls as
they disagree; with weights that sum to one it does not pass $1$
(§12.2), so the missing part, $1 - s$, is "neither". The sign of $c$
is for or against, and $c = 0$ is an even balance. Alec, of the concept
space: "It is in fact a hyperball, but the magnitude is reserved for
certainty." The difference $d = c\,s$ is then a signed value whose size
is lowered by disagreement. Whether $s$ is the certainty he means, or
the size of $c$ is, is his to say.

**Not decided.**

- What Sigma reads: the two lanes (§15), or $d$ and $s$, which to a
  weight matrix are the same thing (each is a sum or difference of the
  other two). The netted value alone does not learn XOR (§15). $c$ and
  $s$ as Sigma's inputs is a different stack and is not tried.
- What a concept keeps where one number is wanted: $c$ (the inverse
  exact, $s$ dropped), or $d$ (the size carries $s$; the inverse is
  approximate unless $s$ is kept beside it).
- Whether $m$ is computed always, or only where a reader asks for the
  pair.

## 17. The comparison of §15, with the unpaired layer (2026-11-14)

Alec: "So please add the unpaired version to the most recent
comparison."

Two stacks are added to §15's. *Unpaired Pi* is today's `PiLayer`
chart under a Sigma that is a weight matrix:
$\tanh\!\big(\tfrac12(\ell_a - \ell_b)\,W + \tfrac12 b\big)\,U + c$.
*Pair, one matrix* is §16's form, the pair from one signed $W$ by two
products, with Sigma reading both lanes. The toy is §15's, run again
(3,000 steps, twenty starts; "fit" is an error under one thousandth;
not the model).

| Task | Hidden | Rectified linear | Unpaired Pi | Pair, one matrix | Pair, two matrices (§15) | The same, half the pairs |
|---|---:|---:|---:|---:|---:|---:|
| XOR | 4 | 13 | 20 | 20 | 20 | 20 |
| XOR | 16 | 20 | 19 | 20 | 20 | 20 |
| Parity of three | 4 | 3 | 17 | 20 | 11 | 0 |
| Parity of three | 16 | 20 | 20 | 20 | 20 | 20 |
| A random truth table of four inputs | 4 | 2 | 11 | 19 | 0 | 0 |
| A random truth table of four inputs | 16 | 20 | 20 | 20 | 20 | 12 |
| $x_1 x_2 + (1 - x_3)/2$ on continuous inputs | 4 | 17 | 20 | 20 | 20 | 20 |
| $x_1 x_2 + (1 - x_3)/2$ on continuous inputs | 16 | 20 | 20 | 19 | 20 | 20 |

Each count is of twenty. (§15 had 12 for parity of three with two
matrices; this run has 11, from a bound added on the input's upper
side.) The rectified stack, the unpaired stack and the one-matrix pair
have the same weights below Sigma at a given width; the one-matrix
pair has twice the weights in Sigma, and the two-matrix pair twice in
both.

Whole stacks (1,024 inputs and outputs, batch 256, forward and
backward, milliseconds; weights in millions):

| Stack | Weights | Processor | MPS |
|---|---:|---:|---:|
| Rectified linear, 1,024 hidden | 2.1 | 1.44 | 0.67 |
| Unpaired Pi, 1,024 units | 2.1 | 1.93 | 0.53 |
| Pair, one matrix, 512 pairs | 1.6 | 2.18 | 0.65 |
| Pair, one matrix, 1,024 pairs | 3.1 | 4.60 | 1.06 |
| Pair, two matrices, 512 pairs | 2.1 | 2.56 | 0.74 |
| Pair, two matrices, 1,024 pairs | 4.2 | 4.45 | 1.20 |

**What it shows.**

1. *The unpaired stack learns these tasks better than the rectified
   one at the same weights and the same cost* on an accelerator (1.3
   times on the processor). The stack is nonlinear because Sigma is a
   plain weight matrix (§4); with Sigma's chart it was linear (§2).
2. *The one-matrix pair is the best of the five here,* and better than
   the two-matrix pair it is a special case of. Claude's reading: a
   signed weight passes through zero freely, so a unit can change the
   lane it takes of an input; two weights each held at zero or above
   cannot, and a unit that starts on the wrong lane stays there.
3. *On inputs at exactly 0 and 1 the unpaired unit is a steep
   two-sided step:* the bound of $10^{-6}$ on the input makes its
   log-odds $\pm 13.8$, so the unit reads the corners through a nearly
   saturated $\tanh$. The pair has no such bound to lean on: a lane at
   0 makes its product 0. The truth tables above favor the unpaired
   stack for that reason, and say less about inputs inside the range.
4. *On the continuous target* the pairs' error is zero to five places;
   the unpaired stack fits under the threshold but not exactly (median
   error 0.0003 at four units), since $\tanh$ of a sum of log-odds is
   not a product of the inputs.

**A correction to §16.** The pair by two products equals the pair of
§15 where no unit takes both lanes of one input, which is the case
when $G^{+}$ and $G^{-}$ are the positive and negative parts of one
$W$. In general
$\ell_a G^{+} + \ell_b G^{-} = \tfrac12\big[(\ell_a + \ell_b)(G^{+} + G^{-}) + (\ell_a - \ell_b)(G^{+} - G^{-})\big]$,
so a unit that takes both lanes of an input needs the second matrix
free, $M \ge |W|$ in place of $|W|$. Taking both lanes of an input
multiplies $z$ and $y$ alike by $(a\,b)^{g}$: it lowers $s$ and leaves
$c$ as it was.

**Not decided** (added to §16's). Whether `PairedPiLayer` is the
one-matrix form (*Claude's recommendation:* yes; it shares `PiLayer`'s
weights, costs one more product, and learned best here), or keeps a
second matrix so that a unit can take both lanes of an input.

## 18. Round D of item 1.8: `SigmaLayer` and `PiLayer` rewritten, a fast `SigmaPiLayer`, and percepts and concepts formed by them (2026-11-14)

The desired changes for a fourth round of item 1.8. There is no text
for Codex yet. Names of round B's source are read in the first
machine's working tree, which is not committed; they are checked again
on the merged source before a text is written.

### 18.1 Alec's words

"PiLayer, which is the unpaired one, seems best in terms of speed and
performance. So let's keeps using it, pull out the unnecessary
tanh/atanh from it and sigma, and make a combined and optimized sigmaPi
layer for use in stacks.

The one thing that needs integration there is the use of the pairs for
regularization. I think the pairs expose the surface that we need to
most effectively drive many of the terms to zero with minimal impact on
accuracy, can you please test that hypothesis and write this to the
spec as the desired implementation changes for a round D in 1.8"

After the test of §18.3:

"With respect to regularization, no need to worry about using the
pairs. But what I want to zero are the use of symbols and concepts;
that's where we want sparsity, not in the weights; so the more zero
outputs the better. That's what to build in D

We want a modification of the LDU form we are using already for the
inverses. Inverses are necessary where we form percepts and concepts
(where we should be using Pi and Sigma).

So don't worry about sparse weights, that's an artifact of sparse
sigma in particular (which is why zero outputs for pi are
significant)."

And: "You don't need to write from scratch, we are talking about
rewriting SigmaLayer and PiLayer."

To Codex (the 1.8 specification, §32.4): "Sigma and Pi should be Layers
that come from the layers.py file."

Then, to three questions (that percepts and concepts are not formed
by the two classes today; whether forming them so is in round D; and
whether a zero output must be exact):

"At the symbolic levels they are not. I think the order0 basic percept
building and zero-order concept building should use pi and sigma,
respectively. They are subsymbolic, so the fit is good there.

3. Let's skip regularization for now, entirely.

So it should be clear now why we need inversion. We are basically just
updating the algorithms used for the existing SigmaLayer and PiLayer.
We are also creating a fast SigmaPi layer that combines both and that
skips the invertibility requirement ( which is the alternate to the
MLP layer, and that works as fast)."

**So round D is three things.** The algorithms of the two existing
classes are updated where they stand, on the LDU matrix they already
have for an inverse. A fast `SigmaPiLayer` combines them with no
inverse, as the model's alternative to an MLP layer. And the basic
percept is built by Pi and the order-zero concept by Sigma, which is
what the inverse is for. There is no `PairedPiLayer` (this replaces
§11 and §15) and no regularization: §18.3 and §18.4 are tests that
were made on the way and are not built.

### 18.2 What is built

**The two classes and their inverse.**

- Each class holds one inner matrix, `self.layer`: a `LinearLayer`,
  or with `invertible=True` an `InvertibleLinearLayer`, which is
  `W = L D U` (`L` unit lower triangular, `d` a diagonal, `U` unit
  upper triangular) with an exact inverse by triangular solves;
  `NonNegativeInvertibleLinearLayer` where `monotonic` is set.
- `PiLayer` is already one entry, the matrix and one exit:
  `_log_mult(x)`, which is `2 atanh(x)`; `self.layer` applied to it;
  `tanh` of half the result. Its `reverse` is the same with the
  inverse matrix.
- `SigmaLayer` wraps its matrix in `_chart` and `_unchart` in
  `forward`, `compute_forward`, `reverse`, `compose`,
  `synthesize_over_set`, `generate` and the butterfly's per-pair
  operations.

**Round B's stacks.**

- `bin/SigmaPiStack.py`, `SigmaPiStack`: `PiLayer(nonlinear=True)` and
  `SigmaLayer(nonlinear=False)` in turn, on `LinearLayer`s, each Sigma
  result clipped to 0..1 by `_field_value`, the last Sigma's matrix
  zero at construction. A field's values, in 0..1, enter Pi through
  its chart, so an absent value (zero) is the product's identity.
- Its users: `expect_stack` and `intuit_stack` (`bin/Spaces.py`; read
  in `ConceptSpread.open_prediction` and `IntuitionField.open_field`)
  and each marker kernel's `stack` (`bin/Language.py`). None inverts
  its stack.

**Where percepts and concepts are formed** (a survey of `bin/` by a
sub-agent, three of its statements checked by Claude).

- Neither goes through a `SigmaLayer` or a `PiLayer`. A percept is a
  codebook row and a greatest or least over its constituents
  (`PartSpace.synthesize_word_parts`, `BasicPercepts.read`). An
  order-zero concept is `ConceptualSpace._compose_order0`:
  memberships raised to the edge weights (`FieldLogic.power`) and
  reduced by the least or the greatest (`fold_presence`), and the
  readout `IndexedSigmaConceptsFromPercepts`, a `tanh` of a weighted
  sum with no inverse.
- The way back is not an inverse: `cs_reverse_presence` and
  `cs_percept_attribution` share a demand out by weight. No
  `_pi_reverse` or `_sigma_reverse` is defined in `bin/`.
- An exact LDU inverse is built in few places, and by the survey none
  is certain to run in the shipped configurations.

**A cost on outputs today.** None is active. `SparsityRegLayer`, a
soft threshold on activations, is built by `PartSpace` and
`ConceptualSpace` with a strength that nothing sets, so it is the
identity. The L1 that is on (`conceptReadoutL1`) is on weights.

### 18.3 A penalty on the weights' terms, tested and set aside

This was the first reading of round D. Alec then set the target to the
outputs (§18.1), so none of it is built; it is kept as the record of
the test.

A *term* is one non-zero weight of Pi: one factor of a unit's product.

**What the pair shows that the unpaired value does not.** With §16's
$r = (\ell_a - \ell_b)\,W$ and $m = (\ell_a + \ell_b)\,|W|$, a unit
with no terms has the pair $(1, 1)$, and every term multiplies
$z\,y$ by $(a_i b_i)^{|W_{ij}|}$, which is at most a quarter to that
power. So the distance of the pair from $(1, 1)$ counts the terms,
each weighed by its input. The unpaired value $c$ is zero both for a
unit with no terms and for a unit whose terms cancel; the pair tells
them apart. Three penalties read from the pair were tried, each zero
for a unit with no terms:

- *product,* $-\log\sqrt{z\,y} = -m/2$. This is a sum of $|W_{ij}|$
  weighed by the input, $-\tfrac12\log(a_i b_i)$.
- *sum,* $-\log\frac{z + y}{2}$.
- *stronger lane,* $-\log\max(z, y) = (-m - |r|)/2$. Its slope in
  $|W_{ij}|$ is $-\log\max(a_i, b_i)$ on a sample where the term
  agrees with its unit's sign and $-\log\min(a_i, b_i)$ where it
  opposes. A term that agrees, on an input that is certain, costs
  nothing; a term on an uncertain input, or one that opposes, is
  charged.

Against them: the sum of $|W_{ij}|$ (*weights*), and the mean of $|c|$
(*unpaired output*).

**The toy** (arithmetic in the session's scratch directory; not the
model and not a receipt). A student $\tanh(\operatorname{atanh}(x)\,W
+ b/2)\,U + d$ of 16 inputs and 12 units (192 terms) is taught by a
teacher of the same form. Task A: the teacher has 15 terms and every
input is spread alike. Task B: the teacher has 36 terms of graded size
(1.5 down to 0.2) and the inputs differ in how far from zero they
reach (0.15 to 0.95). Six starts; medians; error as a share of the
target's variance, on fresh samples.

*First form: the penalty in the loss; then only the k strongest terms
are kept* (the same ranking for every penalty; each penalty at its
best weight for that k, with the weights tried running past the best
on both sides except where marked *):

| Task A, terms kept | 8 | 12 | 16 | 20 |
|---|---:|---:|---:|---:|
| No penalty | 0.541 | 0.146 | 0.0413 | 0.0015 |
| Weights | 0.164* | 0.024 | 0.0017 | 0.0002 |
| Unpaired output | 0.208 | 0.031 | 0.0005 | 0.0001 |
| Pair, product | 0.131* | 0.023 | 0.0015 | 0.0003 |
| Pair, sum | 0.110* | 0.021 | 0.0015 | 0.0003 |
| Pair, stronger lane | 0.113* | 0.021 | 0.0034 | 0.0005 |

| Task B, terms kept | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|
| No penalty | 0.335 | 0.242 | 0.156 | 0.106 | 0.0649 |
| Weights | 0.035 | 0.020 | 0.0083 | 0.0032 | 0.0009 |
| Unpaired output | 0.073 | 0.042 | 0.0191 | 0.0032 | 0.0010 |
| Pair, product | 0.035 | 0.020 | 0.0082 | 0.0028 | 0.0007 |
| Pair, sum | 0.036 | 0.020 | 0.0083 | 0.0024 | 0.0007 |
| Pair, stronger lane | 0.036 | 0.016 | 0.0082 | 0.0021 | 0.0007 |

*Second form: the penalty as a step toward zero after each update, so
that terms become exactly zero and none is removed afterwards* (terms
left, and the error):

| | Weights | Pair, stronger lane |
|---|---|---|
| Task A | 59 terms, 0.000056; 48 terms, 0.00088 | 62 terms, 0.000003; 51 terms, 0.000027; 40 terms, 0.00011 |
| Task B | 66 terms, 0.00017; 40 terms, 0.00083 | 72 terms, 0.00014; 48 terms, 0.00046; 33 terms, 0.0061 |

A third toy, shaped like `expect`'s field (128 lanes, six concepts
present in a sample), did not learn its task under any setting and
tells nothing either way.

**What it shows.**

1. *Any of the penalties is worth a great deal.* At 12 terms the error
   without one is 6 to 12 times that with one.
2. *The pair's penalties are not shown to be better than the plain sum
   of the weights.* In the first form they are equal within the spread
   of the starts on both tasks. In the second form the stronger lane's
   error on task A is 20 to 30 times less at the same number of terms,
   and on task B it is the same.
3. *The pair's penalties are better than a penalty on the unpaired
   output* on task B, by about two times at few terms; on task A the
   unpaired output's penalty is no worse, and better at 16 terms. Task
   B is the one where the inputs differ, and that is the part of the
   hypothesis the toy supports: the pair shows terms where the unpaired
   value does not.
4. So the hypothesis is not confirmed as stated and not refuted. The
   pair's penalty was never worse than the weights' by more than the
   spread, and once much better. Why it should be at least as good
   (*Claude's reading*): it charges nothing for a term that is used,
   so it does not shrink the terms it keeps, and it weighs a term by
   what its input can contribute, so one strength serves inputs of
   different reach.


### 18.4 A cost on the outputs, tested and set aside

Alec: "Let's skip regularization for now, entirely." Kept as the
record of the test, for when it is taken up.

Arithmetic in the session's scratch directory; not the model and not a
receipt. The two classes as they would be after the rewrite, each on
its own LDU matrix: `p = tanh(atanh(x) @ W1)`, `y = p @ W2`, and back
by the two inverses. Sixteen sparse causes (each non-zero one time in
five) are mixed into `x` by the reverse of a hidden layer of the same
form. The student sees `x` only and is taught only by the mean size of
its outputs. Medians of four starts; "near zero" is under 0.01.

| What the cost charges and trains | Outputs near zero | Pi's outputs beyond 0.99 | Distance from the causes | Inverse error |
|---|---:|---:|---:|---:|
| The input, before any teaching | 12% | | | |
| The causes themselves | 80% | | 0 | |
| Sigma's outputs; the diagonals `d` trained | 100% | | 0.080 | 0.0003 |
| Sigma's outputs; the diagonals held | 57% | 80% | 0.091 | 2.0 |
| Sigma's and Pi's outputs; the diagonals held | 73% | 0% | 0.009 | 0.000002 |
| As built today (both layers charted) | 73% | | 0.011 | 0.0000008 |

The inverse error is the largest over the samples whose inputs stay
inside 0.99; at the rails the chart is clipped, as today.

**What it shows.**

1. *The diagonals must not be trained by the cost.* It shrinks them to
   nothing: every output is near zero and nothing is learned (the
   distance from the causes is the size of the causes).
2. *Pi's outputs must be charged with Sigma's.* Charged alone, Sigma's
   outputs are made small by driving four fifths of Pi's to the rails
   and cancelling them in the sum; the inverse is lost there. With
   Pi's outputs charged too, none goes to the rails and the inverse
   holds to two millionths. This is Alec's "zero outputs for pi are
   significant", seen from the other side: an output of Pi that is
   zero is absent from the sum above it.
3. *With both, the layer finds most of the causes it was never shown:*
   near-zero outputs rise from 12% to 73% of an attainable 80%.
4. *The rewritten pair is not better at this than today's charted pair*
   (73% each; 0.009 and 0.011). What the rewrite adds is elsewhere: a
   hidden nonlinearity (§3, §17).

### 18.5 The rules, as decided

Each rule names the section in which it was settled. §18.9 to §18.21
are the record of how; where they and these rules differ, these rules
are the last word. A rule marked *default* is Claude's choice where
Alec has not spoken (§18.8 lists them).

**The two classes, updated where they stand.** *Withdrawn by §18.24,
rule 32* (Alec, 2026-11-16): rules 1 to 4 are not built; `SigmaLayer`
and `PiLayer` stay as they are, and item 1.7's layers are new classes.

1. `SigmaLayer` is its matrix with one exit (§18.16). It calls no
   `_chart` on what it is given, in `forward`, `compute_forward`,
   `_sigma_inner_forward`, `compose`, `synthesize_over_set` and the
   butterfly's per-pair operation. It keeps `_unchart` on what it
   gives (a `tanh`, or `cube_unchart` where `unit_interval` is set).
   `reverse`, `generate`, `generate_functional` and the butterfly's
   reverse undo the exit first and then apply the inverse matrix.
   `nonlinear=False` remains and means the matrix alone, with no
   exit, as the readout and the stack ask for it today.
2. `PiLayer` keeps its values and its one entry and one exit.
   `nonlinear=False`, the unbounded exponential, goes with its
   branches in `forward`, `compute_forward`, `reverse` and
   `factorize_over_set`; no call outside the class passes it.
3. Both keep `self.layer` as it is chosen today (`LinearLayer`,
   `InvertibleLinearLayer`, `NonNegativeInvertibleLinearLayer`), and
   so their weights, their start and their saved names. With
   `invertible=True` each has its exact inverse by the LDU solves.
4. The test that the charted pair is one linear map
   (`TestPiSigmaChartComposition`) is replaced by its opposite: a Pi
   under a Sigma is not an affine function of the charted input.

**The fast layer and its stacks.**

5. `SigmaPiLayer`, in `bin/Layers.py`, is a Pi under a Sigma with no
   inverse (§18.1): two plain weight matrices, the two classes' own
   (`LinearLayer`) with their names. Its `forward` is fused,

   `h = tanh(bounded_atanh(x) @ W_pi + b_pi / 2)`, `y = exit(h @ W_sigma)`,

   with no copy of the activation, no gate, no selection of two
   operands, no butterfly branch and no choice in Python on a tensor,
   so that it is captured as one graph. It has no `reverse`.
6. `SigmaPiStack` is a row of `SigmaPiLayer`s; its keys, its pair
   count and its seed are as round B built them. Every Sigma but the
   last gives `tanh` of its sum (§18.19). *Default:* the last Sigma is
   as round B built it, zero at construction and clipped to 0..1
   (`_field_value` moves into the layer). *Default:* the fused
   `forward` does not compute an inner Sigma's `tanh` and the `atanh`
   of the Pi that follows it, since the two cancel. *Default:* a
   stack's weights start so that a signal keeps its size through the
   pairs; the last Sigma stays zero.
7. Its cost is an MLP layer's (§17). Certificate 7 measures it in the
   model.

**Basic percepts and zero-order concepts, formed by the two classes.**

8. *A basic percept is built by Pi* (§18.14, §18.15). It is a pair,
   for and against, each a product over the unsigned memberships of
   its parts and wholes, in place of the least of
   `BasicPercepts.read` (`minimum`). The layer is a `PiLayer` built
   invertible on the monotone matrix (`monotonic=True`), which keeps
   order over the fold (§18.20). A conjunction is a row of this
   layer: for order zero `store.conjunctive` and the key
   `conceptualPi` go.
9. *A zero-order concept is built by Sigma* (§18.14, §18.18). The
   layer is a `SigmaLayer` built invertible on the signed matrix. The
   sign of a weight chooses the pole it reads: with a percept's pair
   $(p^{+}, p^{-})$ and $W = G^{+} - G^{-}$,

   $c^{+} = p^{+} G^{+} + p^{-} G^{-}$, $c^{-} = p^{-} G^{+} + p^{+} G^{-}$,

   and each lane then takes the exit of rule 1. This is in place of
   the greatest of `ConceptualSpace._compose_order0`
   (`fold_presence`) and of the readout's `tanh` of a weighted sum
   (`IndexedSigmaConceptsFromPercepts`). The concept keeps the pair;
   the signed value, for less against, is a read of it. A zero-order
   concept reads percepts only.
10. *The two matrices are sparse, each an LDU* (§18.11). A row of a
    factor holds entries only for the sources it has been given. An
    entry is written where `PerceptCollections.collect` or a promotion
    writes an edge today (§18.10), at one (§18.18), into capacity
    allocated beforehand. A source newer than its row is an entry of
    that matrix's other triangle.
11. *The way back* (§18.21). From a concept: the exit is undone on
    each lane, their difference returns through `W`'s substitutions
    as the signed percept, and the percept's pair is rebuilt with no
    overlap. The total is not returned. From a percept to its parts
    and wholes: Pi's `reverse`. This is in place of the shares of
    `cs_reverse_presence` and `cs_percept_attribution`. "Both" is
    rendered where the parts are had again, by the forward path.
12. Only order zero changes. The symbolic orders, their definitions
    and their rules are as they are. *Amended by §18.22* (Alec,
    2026-11-16): symbols are split from concepts, as separate tensors,
    and have no dimensions; above order zero the folds keep their
    arithmetic (*put to Alec in §18.23*: rules 8 and 9 at every
    order).
13. There is no regularization (§18.1). §18.3 and §18.4 are tests that
    were made on the way.

### 18.6 Certificates

1. *Not linear.* On random inputs, a `PiLayer` under a `SigmaLayer` is
   not an affine function of the charted input, and the as-built pair
   is (the before case).
2. *The charts that remain.* A test reads the source: `SigmaLayer`
   calls `_chart` only to undo its exit; a `PiLayer` call makes one
   entry and one exit.
3. *The fast layer.* On random inputs over the whole range, the rails
   included, `SigmaPiLayer.forward` equals the `SigmaLayer` on the
   `PiLayer`'s result (atol 1e-6). `expect`'s certificates of round B
   (79 to 82) pass unchanged. Those of the two-pair stacks (`intuit`,
   `marker`) are measured again with their fixtures, seeds and
   presentation counts unchanged, before and after.
4. *Bounds.* For inputs over the whole range, the rails included, and
   weights scaled far beyond their start, every output of each class,
   of the layer and of the stack is finite and in range.
5. *Reverse.* Each class built invertible returns its input (atol
   1e-6) inside the range, for `reverse` and for the set folds and
   their even splits.
6. *One graph.* A stack is captured whole, with no graph break and no
   recompilation across the batch sizes met.
7. *As fast as an MLP layer.* Seconds for a forward and a backward
   call of a `SigmaPiLayer` and of a rectified linear layer of the
   same weights, at `expect`'s size in the lesson configuration, on
   the processor and on the accelerator.
8. *Depth.* A stack of one, two and four pairs learns a stated target;
   the bar is measured first (§18.17).
9. *Formation.* A percept that is for at some parts and against at
   others gives a concept that reads both, with a signed value of
   zero; its way back gives no overlap; and the parts had again give
   both. A percept is read back from its concept, and its parts from
   the percept, to a stated tolerance. A percept collected into an
   older concept's row is read at once, and the inverse holds after.
10. *The lesson.* Its readings before and after rules 8 to 11, each
    order's occupancy, both stopping gates, and one default sweep.

### 18.7 Where it changes values, and what it waits for

- Wherever a `SigmaLayer` is built with its chart: the grammar's
  `lift` where there is no meaning layout (`ChangeLayer._sigma`, and
  `VerbLayer` and `ModifierLayer` by inheritance) and any `sigma` fold
  a grammar file names. Nothing is expected to stay equal there.
- A stack of one pair does not change; a stack of two does.
- Rules 8 to 11 change every reading of order zero.
- It waits for rounds A and B to be merged, since it rewrites
  `bin/Layers.py` and `bin/SigmaPiStack.py`. Rules 8 to 11 touch
  `fold_presence`, which item 1.5 (folds as scatters) and the
  `performance` branch also change; they are planned together.
  *Default:* rules 1 to 7 are built before round C, whose first rule
  reads `expect`. The text for Codex is written then, on the names of
  the merged source.

### 18.8 Claude's defaults, for Alec to overrule

Nothing in §18 waits on a decision. These are the choices Alec has
not made:

- The last Sigma of a stack keeps round B's zero start and clip; the
  fused `forward` skips the pair that cancels; a stack's start keeps
  a signal's size (rule 6).
- Rules 1 to 7 are built before round C (§18.7).
- In rule 9 the exit is taken on each lane of the concept's pair, and
  in rule 11 it is undone on each lane before the difference. Alec
  decided the exit (§18.16) and the pair (§18.18); joining them so is
  Claude's.
- A zero-order concept reads percepts only, so an order-zero row
  defined over another order-zero row has no place at order zero
  (§18.14).
- The way back on an accelerator runs one step for each level of
  dependency in a factor (§18.11); nothing is specified to speed it.

### 18.9 The forming layer at production size (2026-11-14; the single matrix is not built, §18.11)

Alec, asked how the forming Pi and Sigma hold their weights (one
square LDU matrix over round A's order-zero block, or the sparse
store's existing edges): "In production, the percepts and concepts
there will be massive (1M), which is why I was hoping regularization
would allow sparse matrices. Since we don't need explicit access to
the basic percepts other than in these computations, we might want to
use the single SigmaPi since the LDU there are only half full?"

**The single sigma-pi on one LDU matrix.** Pi takes the lower half
and Sigma the upper, with the diagonal between:

`p = tanh(atanh(x) @ L)`, `y = (p * d) @ U`,

and back by `U`'s solve, the division by `d`, and `L`'s solve inside
the chart. `p` is the basic percepts: they are the value between the
two halves, they can still be read there, and each half has its own
exact inverse. It holds half the weights of a `PiLayer` and a
`SigmaLayer` each on its own LDU matrix. As a toy with dense halves,
its inverse held to two millionths inside the range.

**Half full is not enough at a million rows.** A triangle of a million
rows is half of $10^{12}$ weights. The halves have to be sparse
whichever form is taken.

**Regularization cannot be how they become sparse at that size,**
since the dense matrix it would thin cannot be held first. The sparse
pattern has to come from how rows are made: a row holds entries only
for the sources it was admitted with. That is how order zero is
stored today (the readout takes eight references for a concept).

**Sparse and triangular fit.** Order-zero definitions are already
without cycles and are evaluated in the order of their dependencies
(`_compose_order0`); a matrix whose rows depend only on earlier rows
is a triangular one. Rows appended in the order they are admitted
keep it so. A sparse triangular half keeps its exact inverse, by
substitution along its entries.

Checked as arithmetic (not the model): with eight entries drawn for a
row, the inverse of the sparse form is exact to $5 \times 10^{-7}$ at
200 rows; at 1,000,000 rows the two halves are ten million weights,
about 200 megabytes with their indices, and a forward call for four
presentations takes 62 milliseconds on the processor.

**So, proposed** (Claude's recommendation; it joins the two
choices): the forming layer is the single sigma-pi on one LDU matrix
whose halves are sparse and whose entries are the edges order zero
already holds. The rules of formation then read: the basic percepts are `p`,
the order-zero concepts are `y`, and the way back is the two
substitutions.

**Not checked.**

- Who else reads a basic percept's value. Attention's poles and the
  percept forecast may; `p` stays readable either way.
- What the halves can learn when a percept may draw only on later
  inputs and a concept only on earlier percepts, beside its own. The
  toy of §18.4's kind learned under it; nothing wider was tried.
- The way back on an accelerator. Substitution follows the depth of
  the dependencies, so it runs as one step for each level of depth and
  not as one product.
- How a row gains an entry it was not admitted with: §18.10.

### 18.10 Entries grow, by MTM's collection of percepts (2026-11-14; its conflict closed by §18.11)

Alec, asked whether a row's entries are fixed when it is admitted:
"Entries must grow". And: "we should integrate this with the MTM
collection of percepts for concept creation." And, to Codex the same
day (the 1.8 specification, §32.17): "MTM at that stage is collecting
percepts for concept promotion. keeping the idea (LTM from STM) means
keeping the concept, not the percept."

**What is built** (read in the first machine's working tree).
`PerceptCollections.observe(owner, word, percept, occurrence, stream=)`
keeps the percepts witnessed for a word in the pool `owner.mtm`. When
they have recurred `concept_mint_count` times it makes one order-zero
concept, a percept collection (`new_concept`, `pool.promote`). From
then on each further witnessed percept is added to that concept's row
as one more entry (`collect`, through `add_concept_feature`), and the
row's bounds are refreshed (`collection_code`). So today an existing
row does gain entries, and they are for percepts newer than the row.
`FamiliarTier.promote_candidates` makes the other concepts from a
candidate's `support` in the same pool.

**The rule.** The forming layer's entries grow by that collection and
by nothing else: an entry is written where `collect` or a promotion
writes an edge today. Rows and entries go into capacity allocated
beforehand (round A's order blocks), so that no shape changes.

**A conflict with §18.9** (closed by §18.11). In the single LDU matrix a
concept may draw only on its own percept and on earlier ones. A
collection that gains a newer percept needs an entry on the other
side of the diagonal, which is the Pi half. Three ways out, none
chosen:

- the wider collection is a new concept, and the older row is left;
- the rows are kept in an order in which every percept comes before
  every concept that may collect it;
- the forming Sigma is not triangular, and its way back is the even
  split of the set folds, and not a substitution.

**Open.**

- Which of the three.
- What a concept's own percept is: the diagonal ties the concept of a
  row to the percept of the same row.
- The start of a new entry's value. *Default:* one, as `collect`
  writes today.

### 18.11 Decided: two sparse matrices, each an LDU (Alec, 2026-11-15)

"SigmaPi, in order to avoid the ordering issue, can just be two sparse
matrices (each an LDU). We can add the single LDU form to futureWork"

So the question of §18.9 and §18.10's conflict are closed, and §18.9's
single matrix is not built (it is in FutureWork, in the documentation
pass that lands with item 1.8).

**The forming layers.**

1. The basic percepts are built by a `PiLayer` and the order-zero
   concepts by a `SigmaLayer`, each built invertible, each on its own
   matrix `W = L D U` (rules 1 to 3 of §18.5). The rules of formation stand as
   written, with these two matrices.
2. The factors `L` and `U` of each are sparse: a row holds entries
   only for the sources it has been given. An entry is written where
   `PerceptCollections.collect` or a promotion writes an edge today
   (§18.10), into capacity allocated beforehand.
3. A source newer than the row it is given to is an entry of that
   matrix's other triangle. No row is moved and no order of rows is
   required.
4. The way back is each matrix's two substitutions, Sigma's and then
   Pi's inside its chart.

Checked as arithmetic (not the model; 200 rows, four entries drawn for
a row in each factor): the inverse holds to $3 \times 10^{-7}$; an
older row given a newer source, as one entry of `L`, leaves it at
$3 \times 10^{-7}$.

**What follows from a product of factors** (noted, not a question).
The weights are `L D U` and not the entries themselves. An entry added
to `L` changes the weight at its own place and at the places its
target row of `U` reaches: three weights in the check, the entry and
two more. With few entries in a row this stays few.

**Still open, and smaller.**

- What a conjunctive order-zero definition becomes (`conceptual_pi`,
  `store.conjunctive`).
- Whether the two poles of a membership are two inputs or one signed
  value.
- The way back on an accelerator runs one step for each level of
  dependency, in each factor.
- The start of a new entry's value. *Default:* one, as `collect`
  writes today, taken as the weight wanted and written into the
  factor.

### 18.12 What keeps Sigma's output in range (2026-11-15; decided in §18.16)

Alec: "But what will preserve the output to +-1 on the hypersphere? I
don't think there is an easy LDU normalization that keeps outputs in
range."

Pi's exit keeps Pi in range. Sigma, as its matrix, does not: an output
can reach the sum of the sizes of its column of `W`, and no more. So
the only bound that the weights alone can give is that every column's
sizes sum to one or less. Three ways, checked as arithmetic (not the
model; 64 rows, an LDU matrix with about one entry in eight, inputs in
-1..1 of which three in ten are present; the columns' sums ran from
1.6 to 18.7):

| Way | Typical size of an output | In range | Inverse error |
|---|---:|---|---:|
| A scale for each output, from the weights | 0.08 | always | 0.000006 |
| `tanh` on the output only | 0.43 | always | 0.00005; 0.001 where an output is beyond 0.999 |
| A clip | 0.49 | always | 0.000001 where nothing is clipped; lost where it is (18% of outputs) |

- *The scale.* Each output is divided by its column's sum where that
  is above one. It stays an LDU matrix: `W S = L (D S) (S^-1 U S)`,
  and the last factor is still unit upper triangular, so the scale
  goes into the diagonal and the inverse stays exact. The sum can be
  bounded from the factors without forming `W`
  (`(colsum|L| * |d|) @ |U|`, two percent loose here). So there is a
  normalization, and it is cheap; but it makes Sigma a weighted mean.
  One of eight equal sources present reads an eighth, where the
  greatest reads one today, and the typical output here is a sixth of
  the other two ways'.
- *`tanh` on the output only.* `y = tanh(p @ W)`: the first design of
  §2. It is bounded smoothly and has an inverse everywhere, weak in
  single precision near the rails. In one Pi under one Sigma nothing
  cancels it. It would cancel against the entry of a Pi that follows
  it, which the forming pair does not have.
- *The clip* (the first rules, §4). It is the bounded sum: one
  source present reads one. Where it clips, the inverse is lost.

For the length of the vector and not each coordinate (the hyperball):
the weights alone bound it only by one scale for the whole matrix (its
largest singular value), which shrinks more; or the vector is drawn
back to length one where it is longer, which keeps its direction and
loses the excess.

**Claude's recommendation.** Two cases, by whether an inverse is
needed:

- the invertible `SigmaLayer`, which forms the concepts: `tanh` on the
  output only. It is the one way of the three that keeps the range,
  the inverse and a present source's full reading together;
- the fast `SigmaPiLayer`, which has no inverse: the clip, as round B
  built it, so that a Sigma feeding the next Pi is not cancelled.

This would amend the rule for `SigmaLayer` in the invertible case: one `tanh`
at Sigma's exit and none at its entry.

### 18.13 Three smaller points, put to Alec (2026-11-15; answered in §18.14 and §18.18)

**1. A conjunctive order-zero definition.** Today, with the key
`conceptualPi` on, an order-zero concept that has parts written in
`store.conjunctive` is first the least over those parts (all must be
present), and the greatest over its other sources is taken on top
(`_compose_order0`). With concepts formed by Sigma, a sum, there is
no place for "all must be present" in the concept's own row.

- *It becomes a percept* (Claude's recommendation). "All of these" is
  a product, so the conjunction is a row of the Pi layer, and the
  concept reads that percept through Sigma like any other. Order zero
  then has one rule: products below, sums above. `store.conjunctive`
  and `conceptualPi` go for order zero.
- *It stays as built,* a least taken beside the two new layers, with
  the way back it has today.

**2. The two poles.** A membership is a pair, evidence for and
evidence against, each in 0..1, and the two are never netted (Alec,
2026-10: two lanes). Today one set of weights serves both: the
against pole of a fold is the same fold with the source lanes swapped
and least exchanged with greatest (`fold_presence(dual=True)`).

- *Two inputs, one set of weights* (Claude's recommendation). Each
  lane is its own input in 0..1, as round B's stacks take them; the
  against lane of a product is formed with the same entries by the
  dual rule, as today. A row is written once.
- *Two inputs, weights of their own.* The against lane has its own
  entries and is taught apart. Twice the entries; a concept can then
  be denied by sources other than those that assert it.
- *One signed value,* for less against, in -1..1, which is Pi's own
  chart. It nets the lanes, so "both" and "neither" read alike.

**3. The start of a new entry.** `collect` writes a weight of one
today, so one witnessed percept of a collection reads the concept as
fully present.

- *One, as today* (Claude's recommendation). The concept reads as it
  does now from the moment the percept is collected; teaching moves
  it from there.
- *The collection's agreement.* The share of its readings that agreed
  (`observe` keeps it), so a contested collection starts lower.
- *Zero.* Nothing changes when the entry is written; it counts only
  once it has been taught.

### 18.14 Point 1 answered (Alec, 2026-11-15)

"we have basic percepts, formed as a PI, which is a (sparse) hypercone
on hypercube parts and wholes (thus an approximation of the discrete
collection that we were creating with the mereology). Then those
percepts are collected via a (sparse) hyperplane , via the SIGMA,
which become zero-order concepts (or basic concepts)."

So a conjunction is Pi's and is made at the percept: a basic percept
is a product over parts and wholes, and it stands for the collection
the least made before. A zero-order concept (a basic concept) is a
weighted sum of percepts and has no conjunctive definition of its
own. §18.13's first choice is taken: for order zero,
`store.conjunctive` and the key `conceptualPi` go, and what they held
is written as a row of the Pi layer.

*What follows, as Claude reads it.* A zero-order concept reads
percepts only. Today `_compose_order0` also lets one order-zero row
be defined over another and evaluates them in the order of their
dependencies. In two layers there is no such chain: a definition over
other concepts is either written over those concepts' percepts, or it
is a concept of a higher order. Whether any lesson has such a row was
not checked.

Points 2 and 3 of §18.13 are open.

### 18.15 Points 2 and 3, Alec's first answers (2026-11-15; settled in §18.18)

**Point 2.** "We have positive and negative evidence for a basic
percept. Since all the percepts it collects are unsigned, it should
preserve two lanes as a signal of heterogenous input, which means a
pair of percepts. I think it is fair to sum them (pos, neg) to create
the single signed concept: we can observe the percepts to determine
if the uncertainty of the concept was in fact due to mixed evidence
(both rather than neither). Maybe from that point on we do standard
fuzzy kleene logic, I think thats what we had been doing, but if not
lets keep carrying the pair in concepts also."

- *Decided.* A basic percept is a pair, for and against, each formed
  by Pi from unsigned parts and wholes.
- *What had been done at concepts.* Both. The field and the knowing
  hold a pair for every concept at every order (`_observed_knowing`
  and a field's evidence end in two lanes; `_compose_order0` folds
  each pole apart), and Alec's rule of 2026-10 was that the two are
  not netted. The scalar readout
  (`IndexedSigmaConceptsFromPercepts`) is the one place that gives a
  signed value, with zero for unknown.
- *So, by his condition,* the concept keeps the pair. Sigma forms each
  lane; the signed value, for less against, is read from the pair
  where a reader wants the three-valued logic, and "both" is told
  from "neither" by the pair or by the percepts.
- *Open: what the against lane of a concept is.* Today it is the dual
  fold: a collection is denied only when all of its members are
  (`fold_presence(dual=True)`). Summed like the for lane, as a
  hyperplane, one member denied counts against the concept though
  another member is present.

**Point 3.** "A new percept entry? if you mean for inclusion in the
field of the PI, why not let its distance from the center determine
that, since we are recognizing it with a cone? but we did previously
find that a centroid was biased by far-away wholes..."

- *What was asked.* The entry `PerceptCollections.collect` writes: a
  newly witnessed percept joining a concept's collection, which is
  Sigma's row. It is written at one today.
- *Alec's proposal, as Claude reads it, for either layer:* the start
  of an entry is how near the new member is to what the row already
  recognizes. To keep clear of the centroid's bias, the measure is
  the membership the row already gives the new member against its
  bounds (its common part and its covering whole:
  `BasicPercepts.read_points`, `read_collection`), which
  `PerceptCollections.observe` computes today as its `agreement`, and
  not a distance from a mean. *Claude's reading* is that this bias is
  why a collection's code is its bounds today (`collection_code`) and
  not a centroid.
- *Not decided* until Alec confirms that reading.

### 18.16 Sigma's range, decided (Alec, 2026-11-15)

"sigmas range can be as you said, which means its exactly what we
already have."

§18.12's recommendation is taken. Against what is built:

- *The fast `SigmaPiLayer`:* exactly what round B built. Sigma is its
  matrix (`nonlinear=False`) and the stack clips its result.
- *The invertible `SigmaLayer`:* what is built less one thing. Today
  it has `_unchart` at its exit and `_chart` at its entry, and the
  entry is what cancels the exit of the Pi below it, so that a percept
  layer under a concept layer is one linear map in the chart. The
  exit stays; the entry goes.

So the rule for `SigmaLayer` (now §18.5, rule 1) is: `SigmaLayer` calls no `_chart` on what
it is given; it keeps `_unchart` (a `tanh`, or `cube_unchart` where
`unit_interval` is set) on what it gives, and its `reverse` takes the
inverse of that first and then the inverse matrix. Certificate 2
reads: no `_chart` at Sigma's entry, one exit. Where a caller wants
the matrix alone, as the stack does, it asks for it as today.

### 18.17 Sigma's range over several pairs (2026-11-15)

Alec: "So please verify if that sigma range limiting will combine well
over multiple sigmaPi layers"

A toy (not the model): one to six Pi-under-Sigma pairs in a row, 16
inputs in -0.9..0.9, width 32, taught a fixed sum of eight products of
two or three inputs; 3,000 steps; the median of five starts; error as
a share of the target's variance on fresh samples. The weights start
within $\sqrt{3/\text{inputs}}$, which keeps a signal's size.

| Sigma's range kept by | 1 pair | 2 pairs | 4 pairs | 6 pairs |
|---|---:|---:|---:|---:|
| `tanh` at its exit | 0.008 | 0.010 | 0.013 | 0.013 |
| A clip at 1 | 0.003 | 0.147 | 0.915 | 0.970 |
| A clip at 0.9 (three starts) | | 0.034 | 0.171 | |
| A plain `tanh` network, as many matrices | 0.004 | 0.004 | 0.006 | 0.009 |
| A rectified linear network, as many matrices | 0.031 | 0.028 | 0.051 | 0.050 |

**The exit `tanh` combines well.** It learns at every depth tried. The
reason is also its limit: a Sigma's exit and the entry of the Pi that
follows it cancel, so the row is exactly a `tanh` network in which
each Sigma's matrix and the next Pi's are multiplied together
(checked: equal to $6 \times 10^{-7}$). Only the first Pi is a product
of the inputs; the Pi layers after it act on sums. Its inverse stays
exact in principle; in single precision the median error grows from
$4 \times 10^{-7}$ at one pair to $5 \times 10^{-5}$ at six, the worst
sample from $3 \times 10^{-5}$ to $5 \times 10^{-3}$.

**The clip does not.** It is the best of all at one pair and fails
from two. A clipped output sits on the rail, where the next Pi's entry
is at its bound; that Pi then gives values on the rail, and the next
Sigma clips more. At four pairs 62% of Pi's outputs were beyond 0.99
and 42% of Sigma's were clipped, and a clipped output passes no
gradient. Clipping at 0.9 helps and is not enough; passing the
gradient through the clip made it worse (0.48 to 0.95).

**The start matters apart from this.** With the two classes' own
start (within 1 over the inputs) every network here, the two
references included, learned nothing at six pairs; the signal shrinks
at each layer.

**What it means for round D** (not decided).

- The forming layers are one pair, so either way serves them, and
  §18.16 stands.
- A stack of more than one pair: the clip as round B built it is the
  doubtful case. `intuitLayers` and `markerLayers` are two by default
  and `expectLayers` one. Either a stack's inner Sigmas take the exit
  `tanh` (it then trains, as a `tanh` network after its first
  product), or stacks are kept to one pair.
- This toy clips at -1 and 1 and uses a plain `atanh`; round B clips
  at 0 and 1 and bounds the slope of the entry. Whether the two-pair
  clip is why `intuit` and `marker` measured poorly (the 1.8
  specification, §32.14) is not established; one measurement of
  `intuit` at one pair would tell.

### 18.18 Points 2 and 3, decided (Alec, 2026-11-15)

**Point 2.** "Lets consider a soccer ball. we see black and white
hexagons. Therefore we also see non-white and non-black parts. This is
a soccerball basic percept. Now, is it white? is it black? We must,
for those concepts, say "yes and no" (at different places). but
concepts don't work like that; there is no pervasion. so we get a
conceptual 0 value, which means either neither is true or both are
true. So if a concept defines itself in terms of one pole (percept) or
the other, the against lane of a concept comes from the other pole,
and we have concepts that are able to fulfill the both and neither
criteria."

And, of that sentence: ""if a concept defines itself in terms of one
pole (percept) or the other" refers to the sign of the conceptual
vector weight over that perceptual component."

So the pole is chosen by the sign of the weight, and an entry is one
signed weight. With a percept's pair $(p^{+}, p^{-})$ and
$W = G^{+} - G^{-}$, $G^{\pm} \ge 0$:

$$c^{+} = p^{+} G^{+} + p^{-} G^{-}, \qquad
c^{-} = p^{-} G^{+} + p^{+} G^{-}.$$

A positive weight takes the percept's for pole into the concept's for
lane and its against pole into the concept's against lane; a negative
weight crosses them. This is the crossing of §15, in a sum. The
concept keeps the pair: both lanes high is "both", both low is
"neither", and the signed value is zero for either. As in §16 it is
two products:

$$c^{+} - c^{-} = (p^{+} - p^{-})\,W, \qquad
c^{+} + c^{-} = (p^{+} + p^{-})\,|W|.$$

For order zero the exchange of least with greatest goes
(`fold_presence(dual=True)` does that and the swap of lanes today).

*What follows for the way back* (Claude's note, not checked). The
signed part returns exactly through `W`'s LDU inverse. The total,
for and against together, passes through $|W|$, which is not the
product of the same factors and has no inverse by substitution. How
the total is returned is open: by $|W|$'s own inverse where it has
one, or by the even split.

**Point 3.** "write a 1". A new entry starts at one, as `collect`
writes today.

### 18.19 Stacks of more than one pair, decided (Alec, 2026-11-15)

"stacks of more than one pair give their inner Sigmas the exit tanh
for consistency."

So the rules for the fast layer (now §18.5, rules 5 and 6) say, for a
`SigmaPiStack` of more than
one pair:

1. Every Sigma but the last gives `tanh` of its sum, as the invertible
   `SigmaLayer` does (§18.16), in place of round B's clip.
2. *Default:* the last Sigma is as round B built it: zero at
   construction and clipped to the field's 0..1. A `tanh` there would
   give values below zero, and the cube's exit would not give zero at
   the start.
3. *Default:* the fused `forward` does not compute an inner Sigma's
   `tanh` and the `atanh` of the Pi that follows it, since the two
   cancel: `h_next = tanh(h @ W_sigma @ W_pi_next + b_next / 2)`, as
   two matrix products. It differs from computing both only where the
   sum is beyond the entry's bound (about 7.25 in size).
4. A stack of one pair (`expect`) is unchanged. A stack of two
   (`intuit`, `marker`) changes its values, so certificate 3 of §18.6
   holds for one pair, and for two the certificates of round B are
   measured again with their fixtures, seeds and presentation counts
   unchanged.
5. *Default:* a stack's weights start so that a signal keeps its size
   through the pairs (§18.17 found the classes' own start learns
   nothing at six pairs). The last Sigma stays zero.

### 18.20 The way back for a concept's pair: signed and monotone weights (2026-11-15; decided in §18.21)

Alec, of the total that passes through $|W|$ (§18.18): "I think there
is a version that has signed weights. Would that help? the monotone
versions were designed for percepts, which need to maintain order over
the fold."

The two versions are the inner matrices the classes already choose
between: `InvertibleLinearLayer` (signed) and
`NonNegativeInvertibleLinearLayer` (`monotonic=True`, every weight
zero or above, so that order is kept).

- *Signed.* It is the one a concept needs: a negative weight is what
  crosses the poles. The signed part, $(p^{+} - p^{-})\,W$, returns
  exactly through its substitutions. The total, $(p^{+} + p^{-})\,|W|$,
  does not: $|W|$ is not the product of the same factors.
- *Monotone.* With no negative weight, $|W| = W$, and the total
  returns by the same substitutions as the signed part. But nothing
  crosses, so no concept can be defined on a percept's against pole.
  It fits the percepts, as Alec says, and not the concepts.

Checked as arithmetic (64 rows, sparse factors): for a signed $W$ with
405 negative weights, the signed part returns to $10^{-6}$. The total
also returned, to $2 \times 10^{-6}$, but only by a general inverse of
$|W|$ (which was well conditioned here, 34 against 14 for $W$); that
is not a substitution and is not to be had at a million rows.

So the signed version settles the signed part and leaves the total.
Three ways for the total:

- *It is not returned* (Claude's recommendation). The way back gives
  the signed percept, and the pair is rebuilt with no overlap: the for
  pole is its positive part and the against pole its negative part.
  That is exact wherever the percept was not "both" (to $10^{-6}$ in
  the check) and loses the overlap where it was (9% of entries there,
  0.33 on average). It agrees with Alec's "we can observe the percepts
  to determine if the uncertainty of the concept was in fact due to
  mixed evidence". Going forward the concept still has its total,
  as a reading.
- *A second, monotone matrix for the total,* with weights of its own
  and its own substitutions. Exact both ways, at twice the entries;
  the two matrices must be kept such that no lane goes below zero.
- *The total through $|W|$ by an iterative solve.* The same weights,
  no substitution, and a cost that depends on how well $|W|$ is
  conditioned.

### 18.21 The way back, decided: the ball is grey (Alec, 2026-11-15)

"Oh! you are asking about how we imagine the soccer ball. its grey.
Until we remember that the soccer ball is composed of little hexagons
(either conceptually or perceptually). then we can render the "both"
condition that we could not do previously, and it becomes black and
white."

So §18.20's first way is taken. From a concept alone the way back
gives the signed percept, through `W`'s substitutions, and the pair is
rebuilt with no overlap: a concept that was "both" comes back as
neither, which is the grey ball. The total is not returned and
nothing is built to return it. "Both" is rendered where the percept's
parts are had again, from perception or from other concepts: each
part comes back for or against at its own place, and Pi forms the
percept's pair from them as it does going forward.

*Claude's reading:* nothing new is needed for the second half. It is
the forward path over remembered or reconstructed parts, and round D
adds no rule for it.

With this, nothing in §18 waits on Alec. The text for Codex is written
when rounds A and B are merged, on the names of the merged source.

### 18.22 Symbols without dimensions, split from concepts, in round D (Alec, 2026-11-16)

*Alec's words.* On 2026-11-12: "the dimensionality of symbols should be
zero (they are epistemic points which we do not feel, but which serve to
represent collections)". On 2026-11-16, of the pull that meta concepts
exert: "with the impending part D work that reduces symbol dim to zero,
that propagation would seem to break; there would still be an influence
between conceptual orders, but only as mediated by symbols (which would
not have an influence because they are dimensionless and arbitrary).
That seems quite good, actually". Told that round D as written changes
order zero only (rule 12), and asked which round should take symbols'
dimensions away, he answered: "Add it to round D".

Then, on 2026-11-16, to Claude's proposal to keep the shared row (in the
code a symbol's vector is only its activation times its concept's code,
so the split seemed not to be needed):

"'a symbol's vector is only its activation times its concept's code'.
No, a symbol does not need a vector if it lives in a zero-dimension
space. and that is precisely what decouples consecutive conceptual
spaces: a conceptual space influences its vertically adjacent spaces by
pulling the subs to doms, etc, but that effect will not propagate
through a dimensionless space. It will, however, keep basic concepts in
mereological space.

But if you are saying that symbols and concepts are stored together,
then the vertical mappings must not be as I thought. are they in fact
learned with sparse sigma and pi matrices, as I thought step C
indicated?

The split between concepts and symbols is not future work. So make
symbols dimensionless in round D, and split symbols from concepts. If
you want to make promotion easier, it is fine to make a symbol for
every concept premptively, but they are still separate tensors."

*Decided* (Alec, 2026-11-16):

- A symbol lives in a space of zero dimensions and has no vector.
- In round D symbols are split from concepts, as separate tensors. The
  split of 1.8 §3.19 ("Two kinds of row") is no longer FutureWork.
- A symbol may be made for every concept beforehand, to make promotion
  easier; the two stay separate tensors.

*The vertical maps today*, in answer to his question (read on
2026-11-16 in the working tree that round C is changing; no "step C"
is found in the specifications, and 1.8 §3.19 "What is built now" reads
the same):

- They are learned, sparse Sigma and Pi matrices. The store,
  `ConceptualAttentionLayer` (`W_sigma`, a union), and its
  `conjunctive` (`W_pi`, an intersection) are both `SparseLayer` (COO)
  in `bin/Layers.py`. An edge exists once a promotion adds it
  (`add_edge`; `FamiliarTier.cover_members`, `ConceptIndex`). Each
  matrix's values are one `Parameter`, given to the optimizer
  (`definition_matrices`, `_maybe_rebuild_optimizer_for_csw`) and kept
  nonnegative (`project_parts`).
- Their arithmetic is a fold, not the sum of `SigmaLayer` or the
  product of `PiLayer`: each member's value is raised to its edge's
  weight, and the least (Pi) or the greatest (Sigma) is taken, lane by
  lane (`SparseLayer.fold_presence`; `FieldProjection.close` with
  `FieldLogic.reduce`). Round D keeps it above order zero (rule 12; Alec,
  §18.1: "At the symbolic levels they are not. I think the order0 basic
  percept building and zero-order concept building should use pi and
  sigma, respectively. They are subsymbolic, so the fit is good there.").
- Both join rows of one table. A defined row is a concept and is also
  its symbol (`_csw_row_of`); a row's kinds (`symbol_types`: word,
  object, class and others) are tags on that one row. So a definition
  joins concepts to concepts directly, and the chain of §3.19 (concepts,
  by Pi, to symbols; symbols, by Sigma, to the concepts of the next
  order) is not built.

*The rules it adds to §18.5*, to be checked on the source as round C
returns it, before the text for Codex:

14. *Two kinds of row, in separate tensors.* Concepts and symbols are
    separate tables. A symbol is a Pi over member concepts of one order;
    a concept above order zero is a Sigma over member symbols; a concept
    of order zero is the Sigma over basic percepts of rule 9 and has no
    edge from a symbol. This is System 2's vertical structure (1.8
    §3.19: "symbols are PI(concepts), and concepts are SIGMA(symbols)").
15. *A symbol has no dimensions.* It has no vector, no place, no form and
    no trust content read as a direction. Nothing places it or compares
    it by likeness. It has the value its Pi gives, and its edges.
16. *Between conceptual orders, only activation.* A pull (a sub drawing
    its doms together, a dom its subs) acts within one space and does not
    pass through a symbol, so it does not reach the next conceptual
    order. *Claude's reading* of "It will, however, keep basic concepts
    in mereological space": no symbol stands between basic percepts and
    the concepts of order zero (rule 9), so nothing cuts those concepts
    off from mereological space, where they stay placed.
17. *Each lane is a symbol* (1.8 §3.19): a concept's two lanes are read
    by two passes and never netted. *Under question* (§18.23): with
    lanes read by sign, a concept has one symbol, and its two lanes are
    that symbol's pair.
18. *The vertical maps keep their matrices and their arithmetic.* The
    two sparse matrices stay, learned as now. What changes is what they
    join: Pi goes from concepts to symbols, Sigma from symbols to
    concepts. Above order zero each keeps its fold (rule 12); order zero
    changes by rules 8 to 11.
19. *A symbol for every concept, made beforehand* (*default*; Alec allows
    it "to make promotion easier"). Each concept is given two symbols,
    one for each lane, when its row is allocated, each a Pi over that one
    lane. A class is then promoted by writing a Sigma's edges from
    symbols that exist. A conjunction is promoted as a new symbol, the
    Pi over its members, under a new concept, a Sigma over that one
    symbol (*Claude's reading*, joining 1.8 §3.19's "a promoted field a
    pi" with its "A promoted conjunction is then a concept"). Alec
    confirmed the promotion ("Yes", §18.23); two symbols for each
    concept are under question there.
20. The shared row of a concept and its symbol (the invariant of
    2026-09-14) is given up: they are separate tensors.

*Defaults*, for Alec to overrule:

- *Who makes symbols.* *Decided* (Alec, §18.23): allocation, promotion
  and `interpret`. `interpret` makes the meta concept that joins a word
  and its object (the mapping row `InterpretLayer.define` writes), and a
  concept it makes is given its symbols at once. "make all symbols" of
  2026-11-13 meant meta concepts.
- *Where and when.* Replaced by §18.23: a symbol's `.where`, unique in
  the model, is computed and added when it is read (Alec); each event
  of the symbol tensor carries its own `.where` and `.when` (put to
  Alec).
- *Ideas keep their width.* What STM holds for an idea is a vector in
  conceptual space (an absolute truth holds a concept vector, by the
  one-slot row Alec decided on 2026-10-19), not a symbol. Confirmed
  (Alec, 2026-11-16: "Correct").
- *The order inside round D.* The split is built after rules 8 to 11,
  since order zero is the bottom of the chain. Confirmed (Alec,
  2026-11-16: "Good").
- *Before the text for Codex*, Claude maps, in the source round C
  returns, every reader of a symbol's vector or of the shared row: the
  symbol codebook (`symbolTower`, set by `XOR_symbols`, `XOR_exact`,
  `MM_20M_xor` and `MM_add_verb`), the paired symbol event
  (`ConceptEvidence.decode`), the third input of conceptual space's
  combine, attention's `SPACE_SYMBOL`, the content width of `LTM0`
  (`symbol_dim`), `_csw_row_of`, and the addresses `2*cid` and
  `2*cid+1`.

*Certificates*, added to §18.6:

11. Certificates 57 to 62 of the 1.8 specification, with 57 read as: no
    tensor holds both a concept and a symbol; and 60 as rule 19 has it.
12. *No vector for a symbol.* A test reads the source: nothing reads or
    writes a vector, a place, a form or trust content at a symbol. And
    the value a Sigma gives above order zero is unchanged when the places
    of every other order are rotated (spaces-apart, certificate 1).
13. *Symbols beforehand.* Every concept has its two symbols from its
    allocation, and the promotion of a class makes no symbol.

### 18.23 Alec's answers to §18.22; lanes by sign; place at every order (2026-11-16)

*Alec's words*, in reply to §18.22:

- On who makes symbols: "interpret should make two symbols every time
  it makes a concept, to make that decision easier." And: "Interpret is
  more about the process of creating meta-concepts (I misspoke and
  called these symbols). So my proposition was 'lets let all word-object
  associations be done by interpret(), which will create the
  meta-concept that joins them.'"
- On the fold: "The historical way of understanding a human concept was
  as a class (by Aristotle), and that was formalized into set theory.
  Modern neural networks made that a continuous function called an MLP,
  and I call it a Sigma because it involves a summation over a fuzzy set
  that is weighted by membership. so to replace a fold with a sum is not
  inherently different (the way I understand what the fold was doing).
  We can read the fuzzy membership off the weights. The only different
  thing is that I am taking a product of several fuzzy sets to produce a
  category, not pushing them through a nonlinearity. The net effect,
  since I am selecting a hypercone from a hyperball, is to do an OR (a
  multiplication) off of the selected members. But that cone varies in
  width, so we are effectively saying which members must be present. So
  its a class formation in virtue of saying what members are sufficient
  (sum) and what members are necessary (product). If I am wrong about
  that being a fold in our sense, let me know: my understanding is that
  it is a modern and performant version of such."
- On two symbols for each concept (rule 19): "Since we don't want a
  conceptual explosion when walking up the ladder, we can just treat
  positive and negative weights within the sigma as evidence for and
  against. That will give us the two classes of evidence when we need
  it, but also let the hierarchy propagate quickly when we are not
  looking for contradiction."
- On promotion (rule 19): "Yes, that would seem to be the performance
  way of making new symbols and concepts; regularization instead of MLM
  promotion. But they go hand-in-hand, and either may be used within a
  SigmaPi framework. Maybe MLM creates new concepts/symbols as
  necessary, but the membership is learned by the network."
- On where and when: "I had been thinking that because symbols are a
  type of percept, that we perceive them, but we do not conceive of
  them. But perhaps they are unanalyzable, as 0-D percepts. They should
  be associated with a .where (which is unique within the model), but
  since that it computed it can be added when they are retrieved from
  the symbolic codebook (which is really just a tensor that is written
  by the pi of the previous orders concepts and read by the sigma of the
  next orders concepts). As a result we can say 'all higher-order
  concepts are orthogonal to each other because their basis is in
  points, not a shared space'." Then: "But I have a significant concern
  here that I need your help to think about. The early architecture
  used concepts that had a positional encoding in virtue of the .where
  and that shared perceptual space. With their own space, not
  generalized over the perceptual space, it will not be possible to
  abstract over .where. In other words, basic concepts will need to
  flatten their perceptual membership vector (i.e. and operate not just
  on percept activations, but the entire percept vectors). That being
  the case, will the next order have to do the same thing over symbols
  to abstract over its content?" And, a little later: "perhaps symbols
  have zero-D in perceptual space, but we still add a .where and .when
  so that when flattened sigma can abstract over that ???"
- On ideas' width: "Correct". On building the split after rules 8 to
  11: "Good".

*Decided* (Alec, 2026-11-16):

- *`interpret` makes meta concepts.* Every association of a word and an
  object is made by `interpret`, which makes the meta concept that joins
  them. "make all symbols" (1.8 §3.19, 2026-11-13) meant meta concepts.
  A concept that `interpret` makes is given its symbols at once, as
  allocation gives them (how many is put to Alec below).
- *The symbol tensor holds values, and a symbol's `.where` is computed.*
  The Pi of one order's concepts writes the tensor and the Sigma of the
  next order reads it. A symbol's `.where`, unique in the model, is
  computed and added when the symbol is read; it is not stored.
- *Promotion as rule 19 has it* ("Yes"). *Claude's reading:* "MLM" is
  MTM, promotion by recurrence in MTM (`FamiliarTier.promote_candidates`).
  Today a promotion writes an edge at one and the network learns its
  value (rule 10); nothing but a promotion adds an edge, and there is no
  regularization (rule 13). "Maybe MLM creates new concepts/symbols as
  necessary, but the membership is learned by the network" is kept as a
  direction and adds no rule.
- Ideas keep their width; the split is built after rules 8 to 11.

*The fold, in answer to Alec's question* (Claude):

Alec's reading is right. A Sigma says which members suffice and a Pi
which are necessary, and an edge's weight is the fuzzy membership. A sum
and a product are folds in that sense: one operation taken across the
members. Claude reads "an OR (a multiplication)" as an OR over products,
which is what a Sigma over Pis computes; a product alone is the AND, as
the sentence after it has it.

Today's fold differs from `SigmaLayer`'s sum and `PiLayer`'s product in
its operation, not in its kind. It raises each member's value to its
weight (`FieldLogic.power`: $v^{w}$ for Pi, $1 - (1 - v)^{w}$ for Sigma)
and takes the least or the greatest (`SparseLayer.fold_presence`). Four
differences follow:

1. *Counting.* The least and the greatest read one member, the worst or
   the best. A product and a sum read every member, so shortfalls
   compound and evidence adds up.
2. *Learning.* The least and the greatest pass the gradient to that one
   member; a sum and a product pass it to every member. With the sparse
   matrix product, that is the "performant" part.
3. *Signs.* A sum reads a negative weight as the other pole in the same
   pass (rule 9). The greatest cannot take a negative weight as an
   exponent, and would need a pass for each pole.
4. *Exact truth.* The fold keeps an exact 0 and 1 (`FieldLogic.power`
   does not move them). A Sigma with rule 1's `tanh` exit gives less than
   1 for any finite sum.

*Lanes by sign* (Claude, on Alec's words):

- Rule 9 does this at order zero now: the sign of a weight chooses the
  pole it reads ($c^{+} = p^{+} G^{+} + p^{-} G^{-}$,
  $c^{-} = p^{-} G^{+} + p^{+} G^{-}$). Above order zero the same rule
  would read a symbol's pair. Each member enters by one signed edge, so a
  concept needs one symbol, holding its pair, not two (*Claude's
  reading* of "we don't want a conceptual explosion").
- *Claude's reading* of "propagate quickly": before the exit, for less
  against is one signed product, $(p^{+} - p^{-})\,W$, the difference of
  the pair's two sums. So the quick pass is exact before the exit and an
  approximation after a per-lane exit. It cannot show "both", which Alec
  allows when no contradiction is sought; the pair stays what is stored
  (two-truths §1.1).
- The quick pass needs the sum: the greatest has no one-pass difference.
  So lanes by sign mean rules 8 and 9 at every order. A symbol is made by
  Pi from concepts, as a basic percept is made from its parts; a concept
  is made by Sigma from symbols, as an order-zero concept is made from
  basic percepts. That reverses rule 12's "above order zero the folds
  keep their arithmetic" (Alec, §18.1: "At the symbolic levels they are
  not."). *Put to Alec.*

*Place at every order* (Claude, in answer to Alec's concern):

- At order zero today, the abstraction over place comes from how rows
  are read, not from a concept's vector. A basic percept's row is keyed
  by its parts and wholes, not by its place (`BasicPercepts.reserve`;
  only a value percept adds a place). Each row is read in every bracket
  (`BasicPercepts.read(percepts, brackets, rows)`; `read_percepts`: "Read
  each referenced native percept once, then pool before any fold").
  Order-zero concepts are then computed event by event, with one set of
  edges for every event (`ConceptualSpace._compose_order0`, whose field
  is rows by batch by events by lanes). So the place belongs to the
  event, not to the row. A concept is the same wherever its percepts
  occur, and the attention bracket chooses which places are pooled. It
  is weight sharing and pooling, as in a convolution. Nothing is
  flattened, and nothing in it reads a concept's vector, so the split
  leaves it as it is.
- The same holds above order zero if the symbol tensor keeps the event
  axis: symbols by events by lanes. Each event's `.where` and `.when` are
  computed, from its bracket and the model clock, and added when it is
  read, as Alec has it for `.where`. A Sigma reads each event with the
  same edges, which makes it the same concept at any place and time, or
  reads them pooled over the events that attention or recall selects. So
  the `.where` and `.when` belong to the occurrence. The symbol itself
  stays without dimensions, and no Sigma needs a flattened vector to
  abstract over them.
- The `.where` unique in the model is the symbol's address (today
  `WhereRegistry`'s `symbols` range). It names the symbol. Nothing
  abstracts over it but a class, which lists its members.
- A Sigma that reads one event at a time cannot relate two events (this
  one before that one). That needs a Pi over two events at once, as a
  basic percept is an ordered run of letters, or compose in STM. A
  flattened Sigma would not give it either.
- A concept about a place or a time (first, left, earlier) is one whose
  Pi has a place or time part, as a value percept's row has a place
  today.
- With rules 8 and 9 at every order, a symbol is to the next order what a
  basic percept is to order zero: made by a Pi, read by a Sigma, at each
  event. That is *Claude's reading* of "unanalyzable, as 0-D percepts".
- "all higher-order concepts are orthogonal to each other because their
  basis is in points": true of their definitions. Two concepts overlap
  only by the members both list, never by likeness. The pulls within one
  order still make that order's concepts alike or unlike; that likeness
  stays in the order (rule 16).

*Put to Alec* (2026-11-16):

1. Rules 8 and 9 at every order, in place of the fold above order zero;
   each concept then has one symbol, holding its pair.
2. The event axis on the symbol tensor, with each event's `.where` and
   `.when` added when it is read, in place of a flattened Sigma.

### 18.24 Item 1.7 as decided: Pi selects, the percept vectors are pooled, a dense Sigma collects (Alec, 2026-11-16)

This section is the last word of §18 where it differs from §18.5,
§18.22 and §18.23. It was reached in a long discussion; only its final
form is kept here.

*Alec's words.*

- The form: "Percepts (and symbols) are the PI layer. that doesn't
  change much. However, at the basic level (where perceptual space is not
  zero, as in symbolic space), the (square) pi layer emits a sparse
  vector. That sparse vector is multiplied by the full percept vectors,
  and at the basic level it is pooled, so the resulting dimensionality is
  nPercepts*perceptDim (or nPercepts*1 for symbols). Those are collected
  into concepts by the Sigma's DENSE matrix (matmul by either
  perceptDim*nConcepts or 1*nConcepts)."
- What the Pi does: "I want the PI to be selecting (and weighting) the
  percepts to be passed on the the concept. It finds the hypercones,
  weighted by centrality(?), and then sigma operates on those (optionally
  pooled) PI vectors." And: "we let PI determine membership by using
  regularization, and it tells us which percepts are important."
- Every order: "These can be used uniformly across all sigma/pi levels
  also. And for symbols, which have a zero-D perceptual space, the sigma
  returns to its familiar form (i.e. across activation)."
- Pooling, place and time: "Pooling over percepts happens only for basic
  percepts; it is not warranted in higher orders, where the percepts may
  be discontiguous. also, we are not storing .where and .when (the .where
  gets written on STM read)." Then: ".where and .when are not stored FOR
  SYMBOLS. we still have them in other places."
- The mereological order: "In the mereological tower, a new row is
  guaranteed to be greater than its parts (the mereology is a partial
  order). A sum of byte codes is extreme, you just need a max and an
  increment." And: "The basic percept code needs to obey that
  mereological order: it must be greater than its parts and smaller than
  its wholes."
- Likeness: "'cat' and 'act' do not get paired because they are not
  neighbors in mereological space."
- The weights: "One nice thing about that formulation is that the number
  of sigma weights does not depend on the cardinality of its membership."
- The inverse: "sparse LDU for exact inversion. If we have enough
  concepts, we will still have good inversion. But it does mean that we
  cannot assume the inversion, so we may have to adjust the gradient
  teaching algorithm."
- A penalty on the Sigma's weights: "Im not sure thats true that it needs
  a penalty, can we investigate on implemntation?"
- Signed place and time coordinates: "where and when going negative is
  not a problem if the PI is operating in activation space, is it?"
  (They are not: see rule 23.)

*The rules.* Rules 8, 10 and 13 to 16 and 20 stand except as said here.

21. *The Pi layer is the percepts, and above order zero the symbols.* It
    is square and sparse and works on activations (rule 8: a product
    over the memberships of parts and wholes, on the monotone LDU). A row
    is a hypercone, and its value for an input is that input's centrality
    in the cone. Its output is a sparse vector: one pair, for and
    against, for each percept.
22. *At the basic level the Pi's output multiplies the percept vectors,
    and they are pooled.* With $b_p$ the Pi's value for percept $p$ and
    $v_p$ its full vector (`.what`, `.where`, `.when`), the pooled vector
    of a reading is $x = \sum_p b_p\,v_p$, one for each lane, as wide as
    the perceptual space. (Before the sum the array is percepts by
    width; the Sigma is given the sum.) Pooling is within one reading.
23. *A dense Sigma collects the pooled vector into the concepts of order
    zero:* $y = \mathrm{exit}(x\,U)$, with $U$ of perceptual width by
    concepts, signed. A term's sign chooses the lane it adds to, as a
    weight's sign does in rule 9, so each lane stays at or above zero
    although `.where` and `.when` are signed. This replaces rule 9's
    sparse matrix of concepts by percepts at order zero. A concept has
    as many Sigma weights as the space is wide, whatever its membership.
24. *Above order zero there is no vector and no pooling.* The Sigma reads
    the symbols' activations: symbols by concepts. *Claude's correction*
    of "1*nConcepts": a matrix one wide cannot tell one symbol from
    another, since every symbol would add the same row. It is the same
    formula with a symbol's vector taken as its index. *Default:* it is
    stored sparse, as rule 10 has it; the production chart has about a
    million rows.
25. *Sigma and Pi arithmetic at every order.* The least and the greatest
    of `fold_presence` go above order zero too. This reverses rule 12's
    "above order zero the folds keep their arithmetic", and it answers
    §18.23's first question.
26. *Place and time.* A percept's vector carries `.where` and `.when`. A
    symbol stores neither; its `.where` is written when STM reads it.
    §18.23's second question (an event axis for symbols) is dropped.
27. *A percept's code obeys the order of the mereology:* at or above each
    of its parts, at or below each of its wholes. A new row's code is the
    greatest over its parts and an increment, in place of the sum of its
    bytes' codes (`BytesFallbackEncoder.encode`). *Claude's reading:* the
    greatest over the parts and the least over the wholes, as
    `BasicPercepts.subspace_bounds` has them, are what obey it; a mean of
    the parts is not above each part, so a centroid serves as a percept's
    place and not as this code.
    *Default, from a toy of 2026-11-16* (four two-word sentences labelled
    as XOR, width 104, a dense Sigma fitted by least squares): the
    increment is a small draw keyed to the row, not one amount added to
    every coordinate. With words alone the four pooled vectors have rank
    3 and every output is .5. With a row for each pair they have rank 4
    and XOR is read exactly, unless the parts' codes share no coordinate:
    then the greatest over the parts is their sum, a uniform increment
    adds nothing a Sigma can use (rank 3, outputs .5), and a keyed
    increment restores it (rank 4, exact).
28. *Likeness of vectors pairs nothing* (*default*, from "they are not
    neighbors in mereological space"). A row enters a cone, or two rows
    are taken as neighbours, by the mereology or by a promotion, never by
    the cosine of their codes. Before the text for Codex, Claude lists
    the readers that compare codes by likeness (MTM's matching of a
    candidate's support centroid is to be checked first).
29. *The Pi's membership is learned under a penalty* (this reverses rule
    13 for the Pi). *Default:* the penalty of §18.3, measured first. No
    penalty is put on the Sigma's weights; whether a concept that should
    ignore place and time comes to zero weights on those coordinates
    without one is measured at implementation.
30. *The inverse.* The Pi's sparse LDU inverts exactly. The Sigma's dense
    matrix returns the pooled vector by least squares where there are at
    least as many independent concepts as the space is wide, and not
    otherwise; which percepts made the sum is not returned (§18.21, the
    grey ball). Teaching does not assume an exact inverse. Before the
    text for Codex, Claude lists where it does today.
31. *The Pi of rule 21 is the product of memberships* (Alec, 2026-11-16:
    "Use the membership product, PiLayer2's arithmetic"). For each lane,
    with memberships $m$ in 0..1:
    `z = exp(W @ log(clamp(m, EPS_LOG, 1)) + b)`, and back by the exact
    LDU solve, as `PiLayer2` of `bin/Layers.py` has it. One is its
    identity and zero annihilates, so a row requires the members it has
    weight on. This replaces rule 8's naming of `PiLayer` on the
    monotone matrix: that class multiplies $(1+x)/(1-x)$, for which an
    absent unsigned input is the identity. Rule 2 stands for `PiLayer`
    where the grammar uses it on signed values. `PiLayer2`'s inner layer
    (`ContractiveInvertibleLinearLayer`) keeps its diagonal at one or
    more, the rest at zero or more and the bias at zero or less, so a
    row's value is never above its own reading, and a row with no
    reading of its own does not fire. *Default:* the same arithmetic
    for the symbols above order zero.
32. *New classes, with new names* (Alec, 2026-11-16: "please dont
    re-write the SigmaLayer and PiLayer: create newly named versions").
    `SigmaLayer` and `PiLayer` are not edited, so rules 1 to 4 of §18.5
    are withdrawn and nothing that is built with them changes value.
    `PiLayer2` and `SigmaLayer2` exist already and also stay as they
    are: `SigmaLayer2` is the De Morgan wrap of a `PiLayer2` kernel
    (`1 - pi(1 - m)`), which is not the Sigma of rule 23. *Default
    names:* `PiLayer3`, the square sparse product of memberships of
    rules 21 and 31, with a pair of lanes; and `SigmaLayer3`, the
    pooling of rule 22 with the dense signed matrix and exit of rule 23
    (above order zero, rule 24's matrix over symbols). Rules 5 to 7, the
    fast `SigmaPiLayer` and its stacks, edit neither of the two old
    classes and stand.

*What the form is* (Claude, for Alec's question whether it approximates
the full computation, and whether it generalizes in both spaces).

- The full computation has a weight for each concept, percept and
  coordinate: $y_c = \sum_{p,d} W_{cpd}\,b_p\,v_{pd}$. With one vector for
  each percept, the sum over $d$ is one number, $M_{cp} = W_{cp\cdot}
  \cdot v_p$, so the whole array acts as one matrix of concepts by
  percepts.
- Rules 22 and 23 say $M_{cp} = u_c \cdot v_p$, that is $M = U^{\top}
  V^{\top}$ with $V$ the percepts' vectors: a matrix of rank at most the
  perceptual width. The best approximation of any matrix at that rank
  loses exactly the energy of its further singular values
  (Eckart-Young). So the form is exact where a concept's membership is a
  linear function of a percept's vector, and it degrades smoothly where
  it is not.
- That constraint is the generalization. A new percept has a membership
  in every concept from its vector at once (perceptual space), and every
  percept teaches the one row of weights a concept has (conceptual
  space).
- Its limits, to be measured: no more concepts of order zero can be told
  apart in one reading than the space is wide; unrelated percepts add
  to one sum, so a concept's reading carries the others' share (the
  signed weights are what can take it out); and a conjunction of two
  percepts is read only because the code of the whole that has both as
  parts (rule 27) is not a sum of its parts' codes.

*A toy of both orders, trained on XOR* (Claude, 2026-11-16, at Alec's
word: "please add a symbolic layer to the concepts (using a square PI as
usual) and run XOR"; then: "that nonlinearity after sigma is
important"). Plain tensors with this section's arithmetic, not the
model's classes: word percepts, a square monotone Pi, pooling through
nonnegative codes of width 32, a dense signed Sigma to six concepts, a
square Pi over the concepts as symbols, a Sigma over the symbols to
one concept read as the answer. An absent word is 0. Ten seeds, 3000
steps, a penalty on the Pi weights off the diagonal; no seed chosen.

| Exit after Sigma | Pair of lanes | Pi bias | Solved of 10 |
|---|---|---|---:|
| yes | yes | no | 9 |
| yes | yes | yes | 10 |
| yes | no | no | 8 |
| yes | no | yes | 7 |
| no | yes | no | 2 |
| no | yes | yes | 1 |
| no | no | no | 0 |
| no | no | yes | 0 |

So the exit after each Sigma (rule 1) is what carries XOR, as Alec
says; the pair of lanes helps; four further percept rows for the word
pairs change nothing (9 and 10 of 10 again). The Pi weights off the
diagonal were pruned almost to none (about one kept at order zero,
none at order one), so in this toy the two Pis stayed near the
identity and the conjunction was made by the Sigmas' exits over two
orders, not by a Pi row.

*The same toy with the last Sigma linear* (Alec: "can you make the
last layer linear and check again?"). The Sigma of order zero keeps its
exit; the Sigma over the symbols has none.

| Last Sigma | Pair of lanes | Pi bias | Solved of 10 |
|---|---|---|---:|
| linear | yes | no | 10 |
| linear | yes | yes | 10 |
| linear | no | no | 10 |
| linear | no | yes | 8 |
| tanh | yes | no | 9 |
| tanh | yes | yes | 10 |
| tanh | no | no | 8 |
| tanh | no | yes | 7 |

The exit that matters is the one after the Sigma of order zero. A
linear last Sigma solves XOR at least as often as one with an exit.

*Why the first Pi was not enough in the toy* (Alec: "Why isnt the
initial PI sufficient as a nonlinearity? why do we have to add a
second?"). The toy's Pi is `PiLayer`'s chart, `tanh(atanh(x) W)`. It
multiplies odds, $(1+x)/(1-x)$, not memberships: an absent input, 0,
is its neutral element. One row over two words gives .994 with both
present, .9 with one and 0 with none, so it does not require both. A
product of memberships, `exp(log(m) W)` with $m$ in 0..1 (the form of
`PiLayer2`), gives .81, .009 and 0: an absent input annihilates.

| Pi | Exit after Sigma | Penalty | Solved of 10 | Pi weights kept off the diagonal, order zero |
|---|---|---|---:|---:|
| chart (odds) | no | .001 | 2 | 0.0 |
| chart (odds) | no | 0 | 1 | 3.7 |
| chart (odds) | yes | .001 | 9 | 0.4 |
| chart (odds) | yes | 0 | 10 | 0.8 |
| product of memberships | no | .001 | 5 | 0.9 |
| product of memberships | no | 0 | 7 | 4.6 |
| product of memberships | yes | .001 | 10 | 3.3 |
| product of memberships | yes | 0 | 10 | 4.4 |

With the product of memberships the Pi alone solves XOR in most seeds
with every Sigma linear (Boole's sum of products), and with the exit
it solves all ten and the Pi keeps weights, that is, it selects. Rule 8
says "a product over the unsigned memberships" and named `PiLayer`,
whose chart is a different product; Alec chose the product of
memberships (rule 31).

*Certificates*, added to §18.6:

14. *The pooled form equals the flattened one.* With one row of weights
    for a concept repeated for every percept, the flattened computation
    and rules 22 and 23 give the same values (atol 1e-6).
15. *A new percept needs no new weight.* A percept never presented with a
    concept reads that concept from its vector alone; a percept with an
    unlike vector does not.
16. *The standing gates on the new form.* Both stopping gates pass, XOR
    among them, with rule 9's sparse matrix gone at order zero.
17. *How many concepts are told apart.* At the lesson's size, the share of
    order-zero concepts whose reading exceeds every other concept's on
    their own percepts, and the largest reading a concept takes from
    percepts that are not its own. Reported.
18. *The order of the mereology.* Every percept's code is at or above
    each part's and at or below each whole's, and a new row's code is not
    its bytes' sum.
19. *The inverse, as far as it goes.* The pooled vector is returned to a
    stated tolerance where the concepts are enough, and the error is
    reported where they are not.
20. *Place and time without a penalty.* For a concept taught at several
    places, its weights on the `.where` and `.when` coordinates after
    teaching. Reported.

21. *A row requires its members.* One Pi row with weight on two
    members reads near their product with both present and at the
    floor with one absent; no row's value exceeds its own reading.

*Still Claude's defaults:* one symbol for each concept, holding its pair
(§18.23, on "we don't want a conceptual explosion"); rule 24's sparse
storage; rule 28; the penalty of rule 29. The binding of a letter to its
place by a product, considered in the discussion, is not built.

### 18.25 Signed concepts, unsigned percepts (Alec, 2026-11-16)

"I think this is simpler than we are making it. A concept is on the
signed unit hyperball, +-1, so sigma returns values in that range. thats
evidence for and against the concept. those evidence channels (at
output) are stored as two unsigned values. As they are unsigned,
Percepts (pi) can operate on them. That will return twice as many
unsigned values as the next conceptual stage can handle without
doubling, so they are combined into a signed vector for the next sigma.
So if we werent halving the number of symbols/concepts at each stage
becuase we need fewer at higher orders, we would have regular signed
concepts and doubled unsigned percepts."

**The cycle, at every order.**

1. A Sigma gives each of its concepts a signed value in [-1, 1]: its sign
   is the lane, for or against.
2. The value is stored as two unsigned values, one for each lane.
3. The Pi above it (the percepts at order zero, the symbols above) works
   on unsigned values, so it reads those pairs: twice as many inputs as
   there are concepts, and twice as many unsigned outputs.
4. The next Sigma reads each pair combined into one signed input, so it
   reads as many inputs as there are concepts.
5. So an order has n signed concepts and 2n unsigned percepts or
   symbols, and the halving of the rows from one order to the next
   (`_order_caps`) is what the combining in step 4 meets.

**What it settles.**

- *Two unsigned symbols for each concept,* one for each lane: Alec's "two
  symbols every time it makes a concept" (§18.23). Claude's default of one
  symbol holding its pair (§18.24, "Still Claude's defaults") is
  withdrawn.
- *Rule 24:* the Sigma above order zero reads the symbols' pairs combined
  into one signed input for each concept.
- *Rule 23* stands: one signed weight for each input.

**Asked of Alec, and answered below.** Claude's reading of step 1 is that a concept's
value in one reading is one signed number, whose positive part is stored
on the for lane and negative part on the against lane. Then a single
reading is never "both"; "both" arises only where readings are pooled (a
bracket of several events, a field over places), and "neither" is a value
of zero. The other reading keeps rule 23's present wording: the Sigma
gives two sums, one from its positive weights and one from its negative
weights, each unsigned, so that one reading can be "both".

*Answered (Alec, 2026-11-16):* "Yes, one number". In one reading a
concept's value is one signed number in [-1, 1]; its positive part is
stored on the for lane and its negative part on the against lane. A
single reading is never "both"; "both" arises where readings are pooled,
and "neither" is zero. Rule 23 is amended to this: $y = \mathrm{exit}(x\,U)$
is one signed value for each concept, split into its two parts, and not
two sums by the sign of the weights.

*Checked as a toy* (scratch, `xor_cycle.py`; not the model and not a
receipt): the cycle on XOR with four word percepts, Pi as the product of
memberships under `PiLayer2`'s constraints at both orders, six concepts,
the answer read as the last Sigma's signed number, ten seeds each.

| Free percept rows | Target for "false" | Solved of 10 | Off-diagonal Pi weights kept (order 0, order 1), mean |
| ---: | ---: | ---: | --- |
| 0 | 0 (neither) | 10 | 0.3, 0.2 |
| 0 | -0.9 (against) | 9 | 1.1, 0.1 |
| 4 | 0 (neither) | 10 | 1.2, 0.0 |
| 4 | -0.9 (against) | 7 | 1.0, 0.3 |

A first try that only replaced the Sigma of the older toy, and read the
answer from its for lane alone, solved 3 or 4 of 10: the split at the
answer passes no gradient below zero. Where an answer is read, it is read
as the signed number (*default*).

## 19. Item 1.7 as it stands: the rules to build, in one place (Claude, 2026-11-16; a doc pass)

*As of 2026-11-17 the rules to build are restated in §19.21, which
stands where it differs from §19.1 to §19.20.*

This section is the last word of this specification. Everything before
it is the record of how it was reached. Each rule names where it was
settled; a rule marked *default* is Claude's choice where Alec has not
spoken, and §19.7 lists them. It is built after item 1.8 is finished and
pushed, on the source item 1.8 leaves.

### 19.1 What it builds

Two new layers, a Pi and a Sigma, used in one cycle at every order:
percepts (and above order zero, symbols) are unsigned and are rows of a
square, sparse Pi, a product of memberships; concepts are signed and are
a Sigma over them. At order zero the Pi's values weight the percepts'
full vectors, which are pooled before the Sigma. Symbols are split from
concepts and have no dimensions. The old classes are not edited.

### 19.2 The rules

**A. The layers and the cycle.**

1. *New classes* (§18.24, rule 32). `PiLayer3` and `SigmaLayer3`
   (*default names*). `SigmaLayer`, `PiLayer`, `PiLayer2` and
   `SigmaLayer2` are not edited, and nothing built with them changes
   value.
2. *The Pi* (§18.24, rules 21 and 31). Square and sparse, on unsigned
   values: $z = \exp(W \log \mathrm{clamp}(m, \epsilon, 1) + b)$, with
   `PiLayer2`'s constraints (diagonal at one or more, the rest at zero or
   more, bias at zero or less), so a row requires the members it has
   weight on and never exceeds its own reading; its way back is the exact
   LDU solve. Each percept or symbol is two unsigned rows, one for each
   lane (*Claude's reading* of §18.25, "doubled unsigned percepts").
3. *The Sigma* (§18.24, rule 23, as amended by §18.25). One signed weight
   for each input, signed inputs, and for each concept one signed number,
   $\tanh(x\,U + c)$, in $[-1, 1]$ (the exit of §18.16). It is stored as
   two unsigned lanes, its positive part on the for lane and its negative
   part on the against lane; a single reading is never "both". Where an
   answer is read from a Sigma, the signed number is read (*default*,
   §18.25's toy).
4. *The cycle* (§18.25). At an order with $n$ concepts: $n$ signed
   values, stored as $2n$ unsigned lanes; the Pi above reads the $2n$
   lanes and gives $2n$ unsigned values; each pair is combined, for less
   against, into one signed input; the next Sigma reads $n$ inputs.

**B. Order zero.**

5. *A percept's code obeys the mereology* (§18.24, rule 27): at or above
   each part's code and at or below each whole's. A new row's code is the
   greatest over its parts' codes and an increment (*default:* a small
   draw keyed to the row), in place of the sum of its bytes' codes
   (`BytesFallbackEncoder.encode`).
6. *A basic percept is a row of the Pi* over its parts and wholes
   (§18.5, rule 8, with rule 31), in place of the least of
   `BasicPercepts.read`. For order zero, `store.conjunctive` and the key
   `conceptualPi` go.
7. *Pooling within one reading* (§18.24, rule 22, with §18.25, step 4):
   $x = \sum_p (b^{+}_p - b^{-}_p)\, v_p$, with $b_p$ the Pi's pair for
   percept $p$ and $v_p$ its full vector (`.what`, `.where`, `.when`).
8. *The concepts of order zero* (§18.24, rule 23): the Sigma of rule 3 on
   $x$, with $U$ dense, of the perceptual width by the concepts. It
   replaces §18.5's rule 9, the fold of `ConceptualSpace._compose_order0`,
   the readout `IndexedSigmaConceptsFromPercepts`, and item 1.8's stand-in
   for one signed weight (`SparseLayer._fold_weights`, §58 of the 1.8
   specification) at order zero.

8b. *A concept made at order zero starts from its members* (*default*).
   The order-zero matrix of edges goes, so what promotion,
   `PerceptCollections.collect` and the lessons write today (an edge in
   the feature matrix, `add_concept_feature`) becomes a new column of
   $U$: the sum of its members' vectors, normalized, which the network
   then learns. So a new concept reads its members at once, as a new
   percept reads its concepts from its vector (§18.24, certificate 15).
   The Pi of order zero then holds only the mereology's entries, a
   percept over its parts and wholes (rule 6); no concept writes into
   it. It stays full rank: its diagonal at one or more is what gives
   its exact way back (rule 17). The reduced rank of order zero is the
   Sigma's, at most the perceptual width (§18.24, "What the form is"),
   which is why a column of members' vectors is a good start (Claude,
   2026-11-17, on Alec's question whether rule 8b is a reason to reduce
   the Pi's rank).

**C. Above order zero.**

9. *Symbols are split from concepts and have no dimensions* (§18.22,
   rules 14, 15 and 20). Separate tensors; a symbol has no vector, place,
   form or trust content read as a direction; the shared row of a concept
   and its symbol is given up.
10. *Two unsigned symbols for each concept, one for each lane* (§18.25;
    Alec, §18.23: "two symbols every time it makes a concept"). They are
    made when the concept's row is allocated (§18.22, rule 19), and for a
    meta concept when `interpret` makes it (§18.23). A conjunction is
    promoted as a new symbol, a Pi row over its members, under a new
    concept, a Sigma over that symbol (rule 19).
11. *The Sigma above order zero* (§18.24, rule 24, with §18.25): it reads
    the symbols' pairs combined into one signed input for each concept of
    the order below. No vector and no pooling. Stored sparse (*default*).
12. *Sigma and Pi at every order* (§18.24, rule 25). The least and the
    greatest of `SparseLayer.fold_presence` go above order zero too, and
    item 1.8's `SparseLayer._fold_weights` with them.
13. *Place and time* (§18.24, rule 26). A percept's vector carries
    `.where` and `.when`; a symbol stores neither; its `.where` is written
    when STM reads it.
14. *Between conceptual orders, only activation* (§18.22, rule 16): a
    pull within one order does not pass through a symbol.

**D. Learning and the way back.**

15. *The Pi's membership is learned under a penalty* (§18.24, rule 29);
    *default:* §18.3's penalty, measured first. Above order zero a
    promotion still writes an edge at one into capacity allocated
    beforehand (§18.5, rule 10; rule 10 here); at order zero it writes
    a column of $U$ (rule 8b). The penalty thins the values of entries
    that exist; it does not make the matrix sparse (§18.9).
    No penalty is put on the Sigma (reversed above order zero by rule
    22, §19.10; §19.13 shows why it matters).
16. *Likeness of codes pairs nothing* (§18.24, rule 28, *default*). A row
    enters a set, or two rows are taken as neighbours, by the mereology or
    by a promotion, never by the likeness of their codes. The readers that
    decide by likeness today are §19.4, list 1.
17. *The way back* (§18.24, rule 30). The Pi inverts exactly; the Sigma
    returns the pooled vector by least squares where there are enough
    concepts, and not otherwise. Teaching does not assume an exact
    inverse; where it does today is §19.4, list 2.

**E. The learned maps of System 1, and storage.**

18. *The stacks take the same arithmetic* (*Claude's reading* of Alec's
    "These can be used uniformly across all sigma/pi levels"; Alec,
    2026-11-17: yes, §19.10). `SigmaPiLayer` and `SigmaPiStack` (`expect`, `intuit`, the
    marker of `bin/Language.py`) are built from rules 2 and 3 with no way
    back, fused so that a stack is captured as one graph. Today
    `bin/SigmaPiStack.py` uses `PiLayer`'s chart of odds and a linear
    `SigmaLayer` clipped to [0, 1]. This replaces the formula of §18.5,
    rule 5; rules 6 and 7 (keys, pair count, seed, the last Sigma starting
    at zero, the cost of an MLP layer) stand.
19. *The matrices are tensors on the device* (*default*): the Pi and the
    Sigma above order zero are stored as compressed rows, not as
    dictionaries of edges (`SparseLayer._index`), and grow into capacity
    allocated beforehand.

### 19.3 What these rules withdraw or replace

§18.5: rules 1 to 4 (rule 1 here); rule 5's formula (rule 18); rule 8's
`PiLayer` (rule 2); rule 9's two sums (rule 3); rule 10 for the Sigma
(rules 8 and 11); rule 11 (rule 17); rule 12 (rule 12); rule 13 for the
Pi (rule 15). §18.6: certificates 1 and 2. §18.22: rule 17's question and
rule 18 (rules 10 and 12). §18.23: both points put to Alec (rules 12 and
13). §18.24: "Still Claude's defaults: one symbol for each concept"
(rule 10).

### 19.4 Readers to adapt (read in the source on 2026-11-16)

Read by a sub-agent in the working tree of 2026-11-16, while Codex was
building §57.6 of the 1.8 specification; a few statements were read
again by Claude. Names and lines move with item 1.8; Codex takes these
lists again on its starting source (§19.8, step 1). What becomes of each
is Claude's *default* unless a rule above says otherwise.

**List 1: likeness that decides membership** (rule 16).

| Where | What it decides today | Under rule 16 |
| --- | --- | --- |
| `MediumTermMemory.match_candidate` (`bin/Memory.py`) | an observation joins the open candidate whose value has cosine at least `match_cos` (0.8) with it, after an exact match of support proportions fails | the exact match stays; the cosine fallback goes |
| `FamiliarTier.witnessed_match` | with equal member sets, a candidate is the existing concept if proportions are equal or the centroids' cosine is at least `conceptMatchCos` (0.8) | equal member sets decide; the cosine goes |
| `FamiliarTier.observe_field` (two places) | a remembered context like the current one raises an order; a key whose value is like the used centroid counts as accessed | both by membership: the contexts' or the key's members, not their likeness |
| `ReferenceClose.individuals` | each member goes to the anchor whose meet ratio is greatest | by containment, the members' mereology; reported first |
| `IndependentComponents.SparseDictionary.observe` | a residual direction like an admitted or pending one joins it, or a new component is made on recurrence | reported, not changed in this item: it is perceptual, below percepts |

Kept, since they are not likeness of codes: containment, a meet ratio of
exactly one (`MemoryLink.menu` and `link`, `STMChunks.bind_menu`,
`MediumTermMemory.merge`, `ReferenceClose`'s merge of groups,
`TimelineRead.case_coverage`, `Interpret.recognized_objects`); exact
equality (`cdist` at most 1e-5 in `WordExpectation` and
`Expectation.match_symbols`); a likeness that only feeds a learned score,
a cost or a report; decoding a generated vector to the nearest stored
form (`SentenceUnderstanding.readback_scores`, `SynthesizeLayer.reverse`,
`Basis._snap_content`). Also kept, by Alec's decision of 1.8 §48.11
(answer 1): a new identity's start from the objects its words' nearest
words name (`ConceptPlacement.identity_start`). `TruthLayer`'s cosine
thresholds are outside the training step and are left. The likeness
readers with no caller (`priming_prior`, `BracketKeys._cos`,
`_context_cosine`, `_reverse_derive_words`) are removed.

**List 2: teaching that assumes an exact way back** (rule 17).

The grammar's operators keep `SigmaLayer` and `PiLayer` (rule 1), so the
inverses of the generate walk (`_reverse_of_binary_op`,
`SurfaceLayer.generate`, `_bounded_binary_reconstruction`) stay exact.
What changes is the way back through order zero and between orders:

| Where | Today | Under rule 17 |
| --- | --- | --- |
| `Models.reverseReconstruct` (field path), `FieldReverse.reverse`, `cs_percept_attribution`, `cs_reverse_presence`, `SparseLayer.attribute_presence` | a concept's evidence goes back to percepts by undoing each edge's exponent | the Sigma's least squares to the pooled vector, then the Pi's exact solve; the cost against the native target stays, with the fit's residual reported |
| `Models.reverseReconstruct`, the check that a cost of zero is an exact identity, and the docstrings of `_reverse_body` that say the pairing is exact | exactness asserted | the check and the claims go |
| `SentenceField.reconstruct`, `Examination.run`, `Models.reverseOutput` | `model.reverse` of a field, costed | unchanged in form; they read rule 17's way back |
| `CaseSelection.stage_case_bank`; `refine_over_collected`; `Models._passback_scope_ps` | attribution of evidence to percepts | from rule 17's way back |

**List 3: a symbol's vector, or the row a concept and its symbol share**
(rules 9 to 11).

| Where | Today | Under rules 9 to 11 |
| --- | --- | --- |
| `symbolTower` (`SymbolSubSpace`, a codebook of symbols with vectors of the concepts' width), set in `data/XOR_symbols.xml`, `XOR_exact.xml`, `MM_20M_xor.xml`, `MM_add_verb.xml`, `matrix/MM_20M_xor_noraise.xml` | symbols as vectors | goes, with the key; those configurations are ported |
| `forward_concept_to_symbol` | writes a symbol's row from its concept's code; two addresses for each concept, `2*id` and `2*id+1` | writes the two symbols' values (rule 10); no code |
| `ConceptEvidence.decode`, `cs_forward_content`'s paired symbol event, the SBOW slab (`Models`, near `forward_concept_to_symbol`) | one concept code read for both poles | the concept's code is read as the concept's; the symbol gives an activation |
| `ConceptualCombine` with three streams under `symbolTower`; `Models._reverse_body` unpacks two of the three | a symbol's vector as the third input | goes with the tower |
| attention's `SPACE_SYMBOL` keys (`_addressable_spaces`, `_answer_attention_step`, `PrimedSymbolReader`) | symbol codebook rows as keys | the concepts' codes as keys, where a key is needed |
| `LTM0(content_width=symbol_dim)` | the width of an idea | unchanged (ideas keep their width, §18.22), renamed to the concept's width |
| `WhereRegistry`'s `symbols` slice, two for each concept row | two addresses for each concept | kept: two symbols for each concept |
| `_csw_row_of` (`('shared', cid)`, about 110 call sites) and the readers of `('sym', cid)` (`MemberRead.read`, `ClauseRow.concept_point`, `ReferenceClose`, `Queries`, `FamiliarTier`, `Memory.promote`, `Priming`, `Intuition`, `PartClosing`, `PrependedSymbols`, `add_concept_edge`) | one row for a concept and its symbol; a symbol read as a vector | a concept's row and a symbol's index are two lookups; a reader that wants a vector reads the concept's code |

**List 4: the folds and matrices that the cycle replaces** (rules 6, 8,
11 and 12).

| Where | Under |
| --- | --- |
| `cs_read_percept_field`'s fold over the feature matrix (basic-percept literals) | rule 6: rows of the Pi |
| `_compose_order0`'s folds of `store.conjunctive` and of the Sigma store; `IndexedSigmaConceptsFromPercepts` | rules 7 and 8 |
| `_concept_rung_presence` (orders above zero) and `FieldProjection.close` (the closure upward) | rules 11 and 12: the cycle |
| `SparseLayer._fold_weights` and the opposite columns of `_prepare_part_learning` | go: one signed weight needs no opposite column |
| `add_concept_feature` and its writers (promotion in `FamiliarTier` and `Memory.promote`, `PerceptCollections.collect`, `ConceptLessons`, which runs Adam on `features.values`) | rule 8b |
| the key `conceptualPi` (true in `BasicModel_corpus`, `MM_corpus_orders`, `MM_corpus_lessons`, `MM_xor`, `MM_xor_fixture`, `XOR_exact`, `XOR_symbols`; false in `model.xml` and `MM_image`) and its readers | goes (rule 6) |

### 19.5 Certificates

1. *The old classes are unchanged:* `SigmaLayer`, `PiLayer`, `PiLayer2`
   and `SigmaLayer2` give the same values before and after, on random
   inputs over the whole range.
2. *The Pi:* §18.24's certificate 21 (a row requires its members);
   bounds over the whole range, rails included, with weights scaled far
   beyond their start (§18.6, certificate 4); the exact way back (atol
   1e-6, §18.6, certificate 5).
3. *The Sigma:* each concept is one signed number in a reading and its
   two lanes are its parts, never both above zero; bounded as in 2.
4. *The cycle:* two symbols for each concept, made at allocation; the
   next Sigma reads one combined input for each concept below; a class
   promoted makes no symbol (§18.22, certificate 13, amended).
5. *Order zero:* §18.24's certificates 14 (pooled equals flattened), 15
   (a new percept needs no new weight), 17 (concepts told apart,
   reported), 18 (the mereological order), 19 (the way back as far as it
   goes) and 20 (place and time weights, reported).
6. *Symbols:* §18.22's certificates 11 and 12 (no tensor holds both a
   concept and a symbol; nothing reads a vector, place, form or trust
   content at a symbol; a Sigma above order zero is unchanged when the
   places of every other order are rotated).
7. *The stacks:* a fused `SigmaPiLayer` equals its two layers (atol
   1e-6); a stack is one graph; seconds against an MLP layer of the same
   size; depth (§18.6, certificates 3, 6, 7 and 8, on rule 18's
   arithmetic).
8. *Likeness and the way back:* no reader of §19.4's lists 1 and 2 is
   left as it was, each changed as §19.4 says or reported.
9. *The standing gates and the lesson* (§18.24, certificate 16; §18.6,
   certificate 10): both stopping gates pass, XOR among them; the lesson's
   readings before and after, each order's occupancy, the Pi's kept edges
   for each row, and one default sweep.
10. *Speed* (*default*): the seconds of the lesson's sixteen sentences
    before and after, as §51 of the 1.8 specification reported them.

### 19.6 The order of building

1. The two classes, with certificates 1 to 3.
2. Order zero: codes (rule 5), the percept Pi (rule 6), pooling and the
   Sigma (rules 7 and 8), with certificate 5. Decided in §18.22: the
   split comes after order zero.
3. Symbols and the orders above (rules 9 to 14), with certificates 4
   and 6.
3b. The vertical spread (rules 23 to 27, §19.11), with certificates 14
   to 17, 19 and 20.
3c. Thought reads the Sigma and the Pi (rule 28, §19.12), with
   certificate 21.
3d. Certificate 22 (§19.13), on the stacks of step 4 and on one cycle.
4. The stacks (rule 18), with certificate 7.
5. The readers of §19.4 (rules 16 and 17), with certificate 8.
6. Certificates 9 and 10.

### 19.7 Claude's defaults, for Alec to overrule

The names `PiLayer3` and `SigmaLayer3`; each percept or symbol as two
unsigned rows of the Pi; the answer of a Sigma read as its signed number;
the keyed increment of a new row's code; sparse storage above order zero;
the penalty of §18.3; likeness pairs nothing (rule 16); the matrices as
device tensors (rule 19); the speed certificate. Decided since, and no
longer defaults: the stacks on the same arithmetic (rule 18, §19.10).

### 19.8 Codex text (after item 1.8 is pushed)

*Replaced on 2026-11-19 by §22, the item's text in rounds. This text
predates rules 20 to 48 and is not carried.*

~~~
Item 1.7: one structure of Pi, Not and Sigma at every order, and symbols
split from concepts. Read §19.21 of
doc/plans/2026-11-14-sigma-pi-layers.md first: it states the rules to
build, their certificates and their order, and the sections before it
are its record. Build the rules of §19.21 in its order of building, with
its certificates, on main as item 1.8 left it.

Work in /Users/arogers/github/WikiOracle/basicmodel on main. Commit and
push nothing until Alec says so; never push main, and never force-push.
Receipts stay local and are not linked. Do not edit SigmaLayer,
PiLayer, PiLayer2 or SigmaLayer2. The baseline rule of §43 of the 1.8
specification holds: a learning certificate is recorded and not
repaired; a mechanism certificate must pass; both stopping gates must
pass. Do not change a bound, a learning rate, a seed, a capacity or a
presentation count to obtain a pass.

1. Before any edit, take the four lists of §19.4 and the table of
   §19.21 again on the starting source, and return any reader that is
   new, gone or different, with what you propose for it. Where a reader decides by likeness and
   §19.4 does not say what becomes of it, write it with what you
   propose; if that keeps to the specification, build it and go on,
   marked in the return (1.8 §73).

2. Build the order of building of §19.21, steps 1 to 8, each step with
   its certificates, keeping both stopping gates passing after each.
   Where a rule marked default or as Claude's reading in §19.21 cannot
   be built as written, write it with your recommendation; if the
   recommendation keeps to the specification, build it and go on,
   marked in the return (1.8 §73).

3. Port the tests that assert what these rules change; list each with
   its old and new assertion; retire none without naming what replaces
   it. Port the configurations that set symbolTower or conceptualPi.

4. Run once on the finished source: both stopping gates
   (test/test_mm_xor.py::TestMMXorConvergence::test_convergence,
   test/test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class);
   the lesson with its exact fields and its mean difference; the verbal
   proof; the assessment's tables; certificates 9 and 10; and, on this
   finished source only, the item's one default sweep,
   env -u RUN_TRAINED -u RUN_SLOW make test (1.8 §72). Nothing under
   RUN_SLOW. Before the finished source run targeted tests only, and
   name the test files you ran.

5. Return as §20 of doc/plans/2026-11-14-sigma-pi-layers.md: the lists of
   step 1; each step with its edits by file and function and its
   certificates; the ported tests and configurations; the measurements of
   step 4 beside those of item 1.8's last return; the sweep; the source
   snapshot.
~~~

### 19.9 What a model starts with, and what proposes rows and entries (read in the source on 2026-11-17)

*A fresh model.* The lesson's configuration
(`explore-1.xml` of the 2026-11-11 benchmarks), built by
`BaseModel.from_config` with no input, holds 111 concepts, each with its
symbol in today's shared row, and no edge in either matrix:

- 12 at order zero, frozen: the thought predicates the grammar names
  (`implies`, `intuit`, `query`, `exist`, `ask`, `isPart`, `not`, `non`,
  `bind`, `isEqual`, `infer`, `isTrue`), made by `Queries.install`
  through `mint_frozen_concept`;
- 72 at order one: the operators' arguments
  (`prebuild_formal_arguments`, from `PartsOfSpeech.bind`; Alec's round
  C of item 1.8);
- 27 at order two: a class for each operator
  (`prebuild_operator_concepts`).

No word, percept, collection or identity exists before input. Other
configurations with the same grammar: not checked. Under rule 10 these
111 start with 222 symbols. The 12 at order zero have no members, so
rule 8b gives them no start, and they are not basic (1.8 §48.11.2).
*Default,* put to Alec: they stay columns of $U$ at zero, frozen, read by
thought as references and never by perception.

*What proposes rows and entries.* The penalty (rule 15) is on the Pi
only, so what the Sigma gains above order zero is learned, not thinned.

| | Order zero | Above order zero |
| --- | --- | --- |
| Rows of the Pi | a word's basic percept on its first reading, over its parts and wholes (`BasicPercepts.reserve` in the word loop); MTM: a recurring set of percepts becomes a new basic percept over their parts and the wholes they share (`FamiliarTier`) | each concept's two symbols at its allocation (rule 10); MTM: a recurring conjunction becomes a new symbol over its members |
| Concepts, and the Sigma's entries | a word on its first reading (`interpret_word`); MTM: percept collections (`PerceptCollections`) and identities (`FamiliarTier.promote_identity`); lessons. $U$ is dense, so only columns are made (rule 8b) | MTM: classes; every kept reading: an argument gains the word that filled it (`ConceptHierarchy.observe_member`, with no recurrence); `interpret`: meta concepts; lessons; LTM consolidation, with no definition |

### 19.10 Decided: MTM nominates every row of the Pi; one concept for each row; both thinned (Alec, 2026-11-17)

*Alec's words.* "We achieved good simplification of sigma and pi. I dont
like the first-read promotion and no thinning. I'd like to have more
regularity over the architecture, one way is to use MTM to stage
(nominate) basic level percepts and all subsequent symbols, and let
concepts handle themselves (we let pi dictate the number of inputs and
make them square within their sparse matrix)." Asked what is square, and
whether the Sigma is thinned too, he answered "One concept per row" and
"Both thinned".

*The rules.* They amend §19.2 as said, and are built with steps 2 and 3
of §19.6.

20. *MTM nominates every row of the Pi.* A basic percept at order zero,
    and a symbol at every order above it, becomes a row only when MTM
    nominates it on recurrence (`concept_mint_count`); nothing becomes a
    row on its first reading. *Claude's reading:* the tower of parts and
    wholes below the basic percepts is perception and is made on first
    sight, as now. This replaces the basic percept reserved at a word's
    first reading (`BasicPercepts.reserve` in the word loop of
    `bin/Models.py`), the lessons' declared admission
    (`Corpus.admit_training_forms`, `ConceptLessons.admit_native_forms`),
    and rule 10's symbols made at a concept's allocation, with §18.23's
    "interpret should make two symbols every time it makes a concept".
21. *One concept for each row.* Each nominated row brings one concept,
    so every order's Sigma is square: as many concepts as inputs, with no
    halving between orders. A concept starts reading only its own row, at
    one; at order zero that is the column of $U$ given by the row's own
    vector, normalized (rule 8b with one member). Concepts have no
    nomination or promotion of their own: MTM's promotions of classes,
    identities and percept collections become nominations of rows.
22. *Both thinned.* Rule 15's penalty is on the Sigma's entries above
    order zero as well as on the Pi's, measured first. The Sigma of order
    zero is dense (rule 8) and is not penalized. This reverses §18.24,
    rule 29: "No penalty is put on the Sigma's weights".

*Why the split fits* (Claude; checked as arithmetic). In the Pi's
product, an entry from a row to a member read at one changes the row by
$z \log m = 0$ (at 0.9, by $-0.105 z$), so the gradient cannot find a
conjunction from the readings where its members appear together; MTM's
recurrence can. In the Sigma's sum, an entry from an active input changes
the concept by $(1 - y^2)\,x$, which is not zero where the input is
active, so a concept can learn its inputs from use. Rows need nominating;
concepts can handle themselves.

*Claude's defaults, for Alec.*

- *A concept's further inputs.* The one writer of such entries today
  stays: a kept reading adds the word that filled an argument to that
  argument (`ConceptHierarchy.observe_member`; Alec's round C), as an
  entry on the word's symbol once MTM has nominated it, now thinned by
  rule 22. No other writer is added.
- *A new word* is perceived at once, but `interpret` cannot find it
  until its basic percept is nominated; its concept comes with that row.
- *The grammar's declared concepts* (§19.9: 72 arguments, 27 operator
  classes) are declared at construction, each with its row, and are the
  only rows MTM does not nominate. The 12 frozen thought predicates at
  order zero stay the open point of §19.9: they have no percept row.

*Decided* (Alec, 2026-11-17: "A,b,c,d, yes"; of the frozen predicates,
"E nice!"). The three defaults above, rule 18 of §19.2 (the stacks on
the same arithmetic), and §19.9's default for the 12 frozen thought
predicates (columns of $U$ at zero, frozen, read by thought as
references and never by perception) are Alec's.

*Certificates* (added to §19.5).

11. *No row on a first reading.* A new word read once has no basic
    percept and no concept, and `interpret` does not find it; after
    `concept_mint_count` readings MTM nominates its basic percept, its
    concept comes with it, and `interpret` finds it.
12. *Square.* At every order the concepts equal the rows in number, and
    a new concept reads only its own row, at one; at order zero its
    column of $U$ is its row's vector, normalized.
13. *Thinned.* Under the penalty, a Sigma entry above order zero that no
    cost supports goes to zero, as a Pi entry does; the Sigma of order
    zero is not penalized.

The table of §19.9 is what the source does today; rules 20 to 22
replace its first-read and unthinned entries.

*Alec, later the same day.* "When word concepts are interpreted, though,
we need to create the meta-concept structure if it does not already
exist. And that wholes structure cannot be subject to thinning." So:

- *Rule 20, amended.* The meta concept of a word and its object is not
  nominated by MTM: `interpret` makes it, with its structure, when it
  interprets the word and the meta concept does not yet exist. This keeps
  §18.23's "interpret should make two symbols every time it makes a
  concept" for meta concepts.
- *Rule 22, amended.* The meta concept's structure is never thinned:
  neither penalty acts on it. *Claude's reading:* a definition between
  concepts ("balloons are rubber", an edge taught from its `LTM1` part row
  by `DefinitionFold`) stays under rule 22, held by its witness's cost.
- *A word with no object* (Alec: "if we interpret a word without an
  object, we must create a blank one"). Today MTM holds such a word as a
  pending name (`InterpretLayer.reading`, `pool.named`) and its object
  comes later or not at all (after the lesson of 1.8 §58.8, `lives`,
  `moves` and `sleeps` had none). Instead, `interpret` makes a blank
  object, a new concept with no members, together with the meta concept,
  both part of the structure that is never thinned. *Default:* the
  blank object is one order above its word, as an identity is above its
  occurrence. *Its start* (Alec, 2026-11-17), first: "The blank object is
  just a random concept vector." Then, of Claude's first default (near the
  objects of its word's neighbours): "sorry, near the objects of nearby
  words is great. Pure zero is even better, since magnitude is certainty
  in that space." So the blank object's vector starts at zero. *From the
  record:* round C's zero start made recurring contexts alike by place,
  and MTM promoted the same context again until order 2 was full (1.8
  §48.11.6). A blank object is reached through its meta concept, not by
  place, so Claude expects no such effect (not checked). A read that
  normalizes a concept vector must not divide by its zero length. Until it
  gains members it is what 1.8 §48.11.3 called an ungrounded concept.

*Certificate* (added to §19.5). 18. Interpreting a word with no object
makes one blank object and one meta concept, and the blank object's
vector is zero; interpreting it again makes nothing new.

### 19.11 Decided: the vertical spread is built in item 1.7 (Alec, 2026-11-17)

*Alec:* "Build the vertical spread in 1.7", asked where the spread that
rule 3 of 1.8 §55.1 needs is built (1.8 §68.8). What should exist: going
up to containers and down to their missing members (1.8 §3.18, Part A,
approved 2026-11-12); meta concepts primed and priming like any concept
(1.8 §48.11.4); and between orders "an influence between conceptual
orders, but only as mediated by symbols" (Alec, §18.22).

*Checked as a toy* (scratch `spread_down.py`; not the model). A meta
concept reads a word's symbol and its object's, at .8 each; the word is
present and the object absent. Up, the cycle's forward primes the meta
concept (.66). Down, the way back of rule 17 returns the inputs exactly
as they were, so the object gets nothing: an inverse reconstructs, it
does not complete. Handing the meta concept's degree to its members by
their weights gives the object .33. So the spread goes down by the same
matrices read the other way, not by their inverse; this corrects
Claude's reading in 1.8 §68.8.

*The rules* (in the form Alec's answers below gave them).

23. *A fan up and a share down,* on the cycle's own matrices, Pi and
    Sigma alike, by the size of their weights. Up: a concept's or a
    symbol's value is split among the rows that read it, each by its
    weight's share of all the weights it is read with (its fan). Down: a
    row's value is split among its inputs, each by its weight's share of
    the row. What a row receives is summed over all its parents and
    children, at most one; a negative weight passes to the other lane, as
    a weight's sign chooses the lane in rule 3. Down stops at order
    zero's concepts: the Sigma of order zero returns only a pooled vector
    (rule 17).
24. *Siblings get less than the concept that caused them* (Alec). A
    concept present at one gives every other concept the spread reaches
    less than one: what goes up is split by its fan, and of each parent's
    value the concept itself takes back a share.
25. *One pass, written to the expected field at every order.* Up once,
    then down once, from order zero's expectation (`expect`) and the
    observed field. It writes `_expected_knowing` at every order, where
    `ConceptSpread.open_prediction` writes order zero alone today. It runs
    at the open pass, and again at the conversion of 1.8 §55, so that
    `InterpretLayer.preferred_objects` reads what the sentence's words
    primed (*default*).
26. *Not strictly ramified* (Alec). A symbol of order $k$ reads the
    concepts of every order below $k$: "the field of a second order
    symbol is both zero and first order concepts". A meta concept is one
    order above the higher of its word and its object (Alec: "The order
    of the meta concept is good") and reaches both in one step, with no
    symbol between. This amends rule 4, where the Pi above an order read
    only that order's lanes, and §18.22's fields of one order. *Claude's
    reading:* it holds for every symbol, not only meta concepts. A reader
    to adapt is `ConceptAllocator.class_member`, which admits a
    membership only between adjacent orders; step 1 of §19.8 lists the
    others.
27. *Nothing new to learn.* The spread reads the cycle's matrices, adds
    no parameter and no teacher, writes only the expected field, and
    changes no value of the evidence.

*Alec, later the same day.* "Your comments about up and down not changing
the original concept are correct, but children have multiple parents and
parents have multiple children: that is the priming (and we should prime
such that siblings get less activation than the directly responsible
concept)." And: "The order of the meta concept is good; I think it
ultimately points to the fact that human cognition is recursive and not
strictly ramsified. So the field of a second order symbol is both zero
and first order concepts. That will allow the high-order meta concept to
reach both children in one step."

*Checked as a toy* (scratch `spread_siblings.py`; not the model): one
concept present at one, up then down, on 2,000 random sparse graphs of
up to seven parents and nine children.

| Spread | Largest value given to a sibling | Two children sharing five parents |
| --- | ---: | ---: |
| Up by the share of the row present (§3.18), down by row share | 1.000 | 1.000 |
| Up by the cycle's forward, down by the row's largest weight (this section's first form) | 1.000 | 1.000 |
| Up by fan, down by row share (rule 23) | .976 | .500 |

Only the fan keeps every sibling below its cause; the others reach the
cap of one. The division of a source's activation among its links is
the fan effect of spreading activation (Anderson, 1974; cited from
memory, not re-checked).

*Certificates* (added to §19.5).

14. *Up.* A word found raises its meta concept by its share.
15. *Down.* A primed meta concept raises its object when the object is
    absent; the way back of rule 17, on the same case, does not.
16. *Rule 3 of 1.8 §55.1.* Finding a word raises its meta concepts and
    their objects; of a word's two objects with equal expectation, the
    one whose meta concept another word of the sentence primed is
    preferred at the conversion.
17. *Evidence unchanged.* With the spread on and off, every order's
    evidence is equal, and the parameters are the same.
19. *Siblings below the cause.* With one concept present at one, every
    other concept the spread reaches is below one; two children sharing
    five parents of equal weight give each other .5.
20. *One step across orders.* A symbol of order two reads concepts of
    orders zero and one; a meta concept over a word of order zero and an
    object of order one primes the object in one pass, with no symbol
    between.

### 19.12 Decided: thought can read the Sigma and the Pi (Alec, 2026-11-17)

*Alec's words.* "Requirement for 1.7 is introspection of the sigma and
pi. So asking for the definition of cat() would allow the architecture
to query that structure, e.g. parts(cat), wholes(cat) would allow the
architecture to kind word for the parts and wholes of a concept, or the
words corresponding to the symbols below and above the symbol for cat."
*Claude's reading:* "kind word" is "find words".

*What exists* (read in the source on 2026-11-17). Thought already has
the queries `parts` and `wholes` (`bin/Queries.py`: their signatures
return a set, through `_neighbors`), beside `isPart`, `isWhole` and
`define`. They read the part table of LTM1 (`allocator.parts`, through
`Taxonomy.capture_taxonomy`): the relations that sentences stated. They
do not read the Sigma or the Pi. Today the matrices' edges are kept as
dictionaries (`SparseLayer`), which rule 19 replaces with compressed
rows, and §19.4 does not list these queries among its readers.

28. *Thought reads the Sigma and the Pi.*
    - `parts(c)` gives the entries of c's row of the Pi: the symbols
      below c, at every lower order (rule 26), each with its weight.
    - `wholes(c)` gives the rows of the Pi above c that have c as an
      entry, each with its weight.
    - Of the Sigma, thought can read what a concept reads (its own row,
      rule 21, and the inputs `observe_member` adds) and which concepts
      read a given row.
    - A result is a set of references with their weights, not a yes or
      no. A reference's words are those whose meta concepts join them to
      its concept (§19.10); a reference with no word, such as a percept,
      is given as itself.
    - The compressed rows of rule 19 can be read by row, for `parts`, and
      by column, for `wholes` (*default:* an index of columns kept beside
      the rows).
    - *Defaults:* a read is serial and is charged to the work budget as
      the queries are today; it moves nothing (no activation, no weight,
      no row).

*For Alec: what the existing queries read.*
- *Recommended:* `parts` and `wholes` read the matrices, and LTM1's part
  table stays the evidence for `isPart`'s proofs. A stated relation
  reaches the matrices when `DefinitionFold` teaches its edge ("balloons
  are rubber", §19.10), so the queries see it once it is folded.
- *The other way:* they read both, the matrices and LTM1's part table,
  and sum the two as evidence.

*Alec (2026-11-17):* "Ok. We can spec it later." The rule and its
certificate stand; which of the two the queries read is left open.

*Certificate* (added to §19.5). 21. *The structure is readable.* For a
concept c, `parts(c)` and `wholes(c)` give the references and weights of
c's row of the Pi and of the rows above that have c as an entry, with
the words of those that have words; the matrices, the activations and
the rows are unchanged after the read.

### 19.13 Checked: the cycle does and, or and not, and can be read in those terms (2026-11-17)

*Alec's words.* "Please also verify that the SigmaPi stack can do And,
Or, and Not: we had those as separate grammatical operators, and I want
to make sure that what we are doing is interpretable in those terms."

*The grammar's operators* (read in the source). `FieldLogic.reduce` on
the two lanes: `intersection` is the least for and the greatest against;
`union` is the greatest for and the least against; `not` swaps the
lanes; `non` empties both. On one reading, which is one signed number
(§18.25), these are the least, the greatest and the negation.

*How the cycle says them.* A Pi row is a product of the lanes it has
weight on, so it is an *and* of literals: a concept's for lane is the
concept, and its against lane is its *not*. A Sigma is a signed sum of
symbols through tanh, so it is a weighted *or* of those terms, and a
negative weight is a *not*. In symbols (a pair of rows, for less
against), with fa, ga the lanes of a and fb, gb those of b:

| Operator | Symbols, summed by the Sigma |
| --- | --- |
| a and b | (fa fb - ga) + (ga gb - gb) |
| a or b | (fa - ga gb) + (fb - fa fb) |
| not a | (ga - fa) |
| non a | no symbol |

*Checked as a toy* (scratch, `sigmapi_logic.py` and
`sigmapi_readable2.py`, `sigmapi_readable3.py`; not the model and not a
receipt; rows free of the diagonal, as a stack's are).

1. *Exact on true, neither and false.* With the symbols above, the
   Sigma's sum equals the field operator on all nine pairs (and on the
   three values for not) to 2e-6, and after tanh at a gain of 3 to .005.
2. *On graded readings the cycle is the product logic.* Its and is the
   product of the fors, and its or is a + b - ab. Against the least and
   the greatest, over 20,000 random pairs: mean difference .042, largest
   .25 (at .5 and .5: and gives .25 where the least is .5; or gives .75
   where the greatest is .5). Not is exact.
3. *A Sigma alone is not enough.* For and, "neither" at (true,
   neither), (neither, true) and (neither, neither) forces a single
   Sigma's weights and bias to zero. The best single Sigma on a grid
   errs by .76. So and and or need a Pi row.
4. *Learned from random starts* (four symbols, ten seeds, targets at .9):
   and, or and not are solved 10 of 10 with no penalty, with a penalty
   on the Pi, and with a penalty on both.
5. *Read back.* A literal is taken as present when its weight exceeds
   .1; the network with present weights set to one and the rest to zero
   is tested again.

   | Penalty | and | or | not | Literals above .1 (and, or, not) |
   | --- | ---: | ---: | ---: | --- |
   | none | 10 of 10 | 8 of 10 | 10 of 10 | 21.8, 22.8 of 32; 13.2 of 16 |
   | on the Pi only (.003, .01) | 0 | 0 | 0 | 0, 0, 0 |
   | on the Pi and the Sigma (.003, .01) | 10 of 10 | 10 of 10 | 10 of 10 | 4.0, 4.0 of 32; 2.0 of 16 |

   With the penalty on the Pi alone, every Pi weight falls below .1 and
   the Sigma's weights grow to 4 to 8: the answers stay right, but the
   logic is in the Sigma's gain and cannot be read from the rows. With
   the penalty on both (rule 22, Alec's "Both thinned"), each run keeps
   the fewest literals, and unused symbols end with a Sigma weight of
   zero. Seed 0 reads: and as (fa fb - ga) and (1 - gb); or as
   (fb - ga gb) and (fa - 1); not as (ga - fa).

*What follows.*

- The cycle can do and, or and not, and under rule 22 what it learns can
  be read as them. Rule 22 is what makes it readable; rule 15's "No
  penalty is put on the Sigma" would not have.
- A literal that survives the penalty has a small weight (.1 to .3 in
  the toy), because a lane at zero is clamped to 1e-6 and any weight
  gates it. So `parts` and `wholes` (rule 28) read an entry as present
  by a low threshold, not by a weight near one. *Default:* the threshold
  is a key, measured with rule 15's penalty.
- *For Alec, not blocking:* on graded readings the layers and the
  grammar's field operators differ (point 2). They agree on true,
  neither and false. Either they are left to differ, or
  `FieldLogic.reduce` takes the product and a + b - ab so that both say
  the same.

*Certificate* (added to §19.5). 22. *And, or and not.* A stack of rules
2 and 3 learns and, or and not over true, neither and false from random
starts; with rule 22's penalty, the network whose present entries are
set to one and the rest to zero gives the same answers.


### 19.14 Item 1.7 after 2026-11-17: what is decided and what is open (Claude; a doc pass)

Alec: "Please do any prep work necessary for 1.8 and 1.7". §19.2 to
§19.8 are the rules as of 2026-11-16; §19.9 to §19.13 amend them. The
Codex text of §19.8 names them all.

*Decided on 2026-11-17 (Alec).*

- Rules 20 to 22: MTM nominates every row of the Pi; one concept for
  each row, the Sigma square; the Pi and the Sigma above order zero are
  both thinned (§19.10). `interpret` makes a word's meta concept, whose
  structure is never thinned, and a blank object for a word with none;
  the blank object's vector starts at zero.
- The three defaults of §19.10, rule 18, and the 12 frozen thought
  predicates as §19.9 has them.
- Rules 23 to 27: the vertical spread, up by fan and down by share, one
  pass, not strictly ramified (§19.11).
- Rule 28: thought reads the Sigma and the Pi (§19.12).

*Amended in place by this pass.* Rule 15's "No penalty is put on the
Sigma" now points to rule 22. §19.7 no longer lists rule 18 as a default.

*Readers to add to §19.4's lists* (read in the source on 2026-11-17;
Codex takes them again in step 1 of §19.8):

| Where | Today | Under |
| --- | --- | --- |
| the thought queries `parts` and `wholes` (`bin/Queries.py`, `_neighbors`), through `Taxonomy.capture_taxonomy` | read LTM1's part table (`allocator.parts`) | rule 28; which of the two they read is open (§19.12) |
| `ConceptualSpace.concept_parts` and `concept_wholes` (`bin/Spaces.py`) | read the allocator's part and whole references of a concept | rule 28's reads of the Pi, where the matrices replace those references (rules 6 and 10) |
| `InterpretLayer.reading`'s pending name (`pool.named`) and `PrependedSymbols.stage`, which holds it in STM | a word with no object is a pending name in MTM | the blank object of §19.10 |

*Open, none blocking the build.*

1. Which of the two the queries `parts` and `wholes` read (§19.12; Alec:
   "We can spec it later").
2. On graded readings the layers give the product logic and the
   grammar's field operators the least and the greatest (§19.13).
3. The threshold by which an entry is read as present (§19.13,
   *default:* a key, measured with rule 15's penalty).
4. The remaining defaults of §19.7.
5. Whether and and or are made monotone, with not the only crossing of
   the lanes (§19.15): decided in §19.18, as rule 29.
6. Rule 7's sum replaced by the product on the complements (§19.17).
7. The readings of §19.20 (`expectLayers` stays at 1, §19.19).

### 19.15 Monotone and and or; not carries all negation (Alec, 2026-11-17; a direction with its check)

*Alec's words.* "Im unclear on whether and/or should be written to
product logic, or what exactly that entails. I do think they might
benefit from being monotone if that would allow us to also force the NOT
operation to carry all negation. In particular, conceptual constructions
that correspond to counterfactuals should be forced to carry a NOT
operation, rather than having that sign flip masquerade within AND and
OR. So if writing them in product logic forgoes that, then we should not
make them match (and in fact, we should consider changing how intuit and
the rest work so that they need NOT for negation). This is important
because the removal of the NOT is exactly what is necessary to speak
only of the good in Augustine's sense, where evil does not exist except
as the privation of the good." And then: "Please capture this in the
doc, and how it dovetails to Shamatha Speak, the language that speaks in
absolute truth (which can be understood via the tripartite division of
knowledge into what is absolute, what is relative/dependent, and what is
imaginary (this last being augustine's removal))."

*Product logic, and why it is a separate matter.* It is only how two
graded certainties combine. The grammar's operators take the weaker for
and and the stronger for or (`FieldLogic.reduce`); the layers multiply
for and (two halves give a quarter) and give a + b - ab for or (two
halves give three quarters). Both are monotone: raising an input never
lowers the answer. So matching them or not neither gives nor forgoes
what Alec asks. *Default:* they are left to differ (§19.13).

*Where a negation can hide in rules 2 and 3 today.*

1. A negative weight of a Sigma: evidence for an input becomes evidence
   against the concept.
2. A row of the Pi that reads the other lane: a symbol's for row with
   weight on a concept's against lane.
3. A bias below zero: with no evidence at all, the concept is denied.

§18.20 recorded the first as the Sigma's purpose ("a negative weight is
what crosses the poles"), with Alec's "the monotone versions were
designed for percepts". The direction here extends the monotone version
to the concepts and gives the crossing to not alone.

*The monotone cycle.* A symbol's for row reads only for lanes and its
against row only against lanes; a Sigma's weights are zero or more and
it has no bias; not is its own operation, the swap of a concept's lanes
(`FieldLogic.reduce`'s `not`), and nothing else crosses. And and or
remain exact on true, neither and false:

| Operator | Symbols (for row less against row), summed with weights of one |
| --- | --- |
| a and b | (fa fb less ga) and (fa fb less gb) |
| a or b | (fa less ga gb) and (fb less ga gb) |

*Checked as a toy* (scratch, `sigmapi_monotone.py`; not the model and
not a receipt; ten seeds, four symbols, the penalty of rule 22 on both;
"not offered" gives each concept's swap as a further input).

| Task | Cycle | Not offered | Solved | Steps where raising an input lowers the answer | Answers against, given no evidence against |
| --- | --- | --- | ---: | ---: | ---: |
| and | signed (rules 2, 3) | no | 10 | 0% | 0% |
| and | monotone | no | 10 | 0% | 0% |
| or | signed | no | 10 | 0% | 0% |
| or | monotone | no | 10 | 0% | 0% |
| not | signed | no | 10 | 100% | 100% |
| not | monotone | no | 0 | 0% | 0% |
| not | monotone | yes | 10 | 100% | 100% |
| xor | signed | no | 10 | 48% | 100% |
| xor | monotone | no | 0 | 0% | 0% |
| xor | monotone | yes | 10 | 48% | 100% |

So the monotone cycle learns and and or as well as the signed one; it
cannot negate at all unless a not is offered; and with a not offered it
learns what the signed cycle learned. Where no not is used and the input
holds no evidence against, it never answers against: it says for, or it
says nothing.

*What it would change in §19, if it becomes a rule.*

- Rule 3: the Sigma's weights are zero or more, with no bias below zero,
  in place of "one signed weight for each input". A concept still has
  one signed value; its against part comes from its inputs' against
  parts, or from a not.
- Rule 2: a row reads its own lane only.
- Rule 18: `expect` and `intuit` can no longer learn a negation of their
  own; a counterfactual is carried by a not in thought, as Alec asks.
- Rule 23's "a negative weight passes to the other lane" has nothing to
  act on; a not passes to the other lane.
- §18.20: with no negative weight the total returns by the same
  substitutions as the signed part, so the way back of §18.21 can be
  looked at again.
- *Claude's reading:* a crossing is then an entry made by the operator
  `not`, in a sentence or in thought, never one found by the gradient.
  Rule 28 can list every one of them, and they can be left out.
- *Not checked:* whether either stopping gate, or order zero, needs a
  crossing that no `not` supplies. *Expected, not checked:* the XOR
  gates do not, where "true" and "false" are words that are each
  perceived, since then the answer is an or of ands over what is
  present.

*How it dovetails with Shamatha Speech.* What exists:
`doc/Philosophy.md`, "Shamatha Speech and Single-Pointedness", gives it
as the object grammar of ands and ors with a condition that the parts
stay one connected field; `data/shamatha.grammar` is its grammar and
carries a `not` rule today. The same file's account of the three natures
has the dependent (*paratantra*), the imagined (*parikalpita*, "which
the architecture does not build") and the perfected (*parinispanna*),
the dependent seen without the imagined.

Alec's note adds: Shamatha Speech is the language that speaks in
absolute truth, understood by that division into the absolute, the
relative or dependent, and the imaginary, and the imaginary is what
Augustine removes. *Claude's reading, for him to correct:* with not the
only crossing, the language without not is that speech. Its ands and ors
hold only what was witnessed, each dependent on its parts; it can say
for or stay silent, and what is absent is "neither", a privation and not
a thing. A statement against something enters only through not, which
is where the imaginary would be spoken. Removing not is Augustine's
removal (evil as the privation of the good: *Enchiridion* 11,
*Confessions* VII.12; cited from memory).

*For the documentation pass* that lands with item 1.7: this section to
`doc/Philosophy.md`, beside "Shamatha Speech and Single-Pointedness" and
the three natures, with the reference to Augustine checked; and whether
`data/shamatha.grammar` keeps its `not` rule is Alec's. `doc/*.md` is
not edited now (`basicmodel-docs`).

*For Alec.* Make the monotone cycle a rule of item 1.7 (it would be rule
29, with a certificate from the table above, and Codex checking the
stopping gates first), or keep it as a direction for now.

### 19.16 A Sigma-Pi-Not stack; the order at the basic level; one Pi; pooling (Alec, 2026-11-17)

*Alec's words.* "Expect and intuit should be able to learn a negation,
so maybe we can create them with SigmaPiNot. Lets verify the order of
operations at the basic level. DNF? CNF? Right now we have PI to make
basic percepts, then SIGMA to make concepts, and possibly another PI to
make symbols. The two-lane percepts get combined into a signed hyperball
for conceptual space, then separated back out. Also is NOT applied to
concepts, or do we reason over both truth lanes at the level of
percepts/symbols? Since we need only one operator of each kind, Im
thinking that basic percepts should be the single PI, so that we dont do
more work than is necessary for a full logical expression. that
dovetails with MTM doing the promotion to symbols at the next level,
only when they are found in the input. Ive been thinking about the
Pooling that PI is doing. Can we or should we make that the actual
cross-content PI, instead of restricting PI to operate on the entire
vectors?"

**1. A Sigma-Pi-Not stack.** Not is a layer of its own: it offers each
concept's swap. The Pi's rows read their own lane, and the Sigma's
weights are zero or more with no bias (§19.15). An entry that reads a
swap is a Not entry, and is penalized as the others are.

*Checked as a toy* (scratch, `sigmapi_not_count.py`; not the model; ten
seeds, four symbols, true, neither and false):

| Task | Solved | Entries read directly | Entries read through Not | Runs using no Not |
| --- | ---: | ---: | ---: | ---: |
| and | 10 | 4.0 | 0.0 | 10 |
| or | 10 | 4.0 | 0.0 | 10 |
| not | 10 | 0.0 | 2.0 | 0 |
| nand | 10 | 0.0 | 4.0 | 0 |
| xor | 10 | 4.0 | 4.0 | 0 |

So a stack so built learns a negation, and a Not appears exactly where
the function needs one, in the fewest entries. *Recommended:* `expect`,
`intuit` and the marker are built as Sigma-Pi-Not stacks (rule 18).

**2. The order at the basic level** (as rules 2 to 8 have it).

| Step | Operator | What it gives |
| --- | --- | --- |
| 1 | Pi (rule 6) | a basic percept: an and of its parts and wholes, on each lane |
| 2 | pooling (rule 7) | each percept's pair, for less against, times its vector; summed over the percepts of one reading |
| 3 | Sigma (rule 8) | a concept of order zero: one signed value, an or over the percepts, weighted through their content |
| 4 | the split | the signed value stored as two lanes |
| 5 | Pi, above | a symbol: an and over concepts' lanes, only where MTM nominated a row |
| 6 | Sigma, above | a concept of the next order |

- *The basic level is an or of ands:* disjunctive normal form. The next
  order reads an and of ors over it, so the alternation gives both
  forms. With Not on the concepts, a block is Not, then Pi, then Sigma:
  negation at the literals, then and, then or, which is a full logical
  expression.
- *Both lanes at once.* Within a block the two operators act on the two
  lanes dually: for a and b, the Pi makes the for lane (the product of
  the fors) and the Sigma makes the against lane (the sum of the
  againsts); for a or b, the other way round (§19.15's table).
- *Where Not is applied.* Under rules 2 to 4 as written, both: the Pi
  above reads all 2n lanes, so a row can take a concept's against lane
  as a literal, and the Sigma's weights are signed. The two are one
  arithmetic: a for row that reads a concept's against lane reads that
  concept's Not. Under §19.15, Not is applied to concepts, as the swap
  of a concept's pair between a Sigma and the next Pi; the Pi and the
  Sigma on either side stay within their lanes. The percept Pi, below
  the first Sigma, has no Not: perception gives evidence against only
  where the input does (in the lesson of 1.8 §58.8 all 139 words read
  (1, 0)).

**3. One Pi.** The percept Pi, the Sigma and Not on its concepts are a
full expression, so no second Pi is computed as a layer. This is what
rules 20 and 21 already give: above order zero a row exists only where
MTM nominated it on recurrence. Rule 4's "the Pi above reads the 2n
lanes and gives 2n unsigned values" described a full Pi at every order;
it is read as the nominated rows only. A conjunction of percepts found
in the input is a basic percept that MTM nominates (§18.24, rule 27's
toy: with a row for each pair, XOR is read exactly).

**4. Pooling.** *What exists.* Alec, 2026-11-16 (§18.24): "I want the PI
to be selecting (and weighting) the percepts to be passed on the the
concept"; its output "is multiplied by the full percept vectors, and at
the basic level it is pooled". So the Pi gives one pair for each
percept, and the pooling is a sum of whole vectors.

*Claude's reading of the question, for Alec to correct:* a product
across the percepts' content, coordinate by coordinate, in place of the
sum of whole vectors. *Checked as a toy* (scratch, `pooling_rank.py`;
four word percepts, four two-word sentences labelled as XOR, width 104,
a dense Sigma fitted by least squares, as rule 27's toy):

| Codes | Pooling | Rank | Largest XOR error |
| --- | --- | ---: | ---: |
| share coordinates | sum | 3 | .5 |
| share coordinates | product | 4 | 0 |
| share no coordinate, absent as zero | product | 1 | .5 |
| share no coordinate, absent as one | product | 3 | .5 |

- With codes that share coordinates, the product makes a conjunction of
  two percepts readable with no row for the pair. With codes that share
  none it gives nothing.
- On codes that obey the mereology (rule 5), a product by coordinate is
  what two percepts share, and the sum is everything present.
- `.where` and `.when` are signed (sines and cosines), and the product
  needs unsigned values; Alec, 2026-11-16: "where and when going
  negative is not a problem if the PI is operating in activation space".
- It would be a second way to conjoin, beside the nominated rows of the
  one Pi (point 3).
- *Recommended, on this reading:* the pooling stays a sum, and
  conjunctions stay rows of the Pi.

*For Alec.* (a) Whether §19.15 and point 1 become rule 29 of item 1.7:
monotone rows and weights, Not as a layer on concepts, and the stacks
built as Sigma-Pi-Not. (b) Whether the reading of point 4 is the one
meant.

### 19.17 Unsigned place and time; the pooling as a multiplication; where Not sits, and System 1 (Alec, 2026-11-17)

*Alec's words.* ".where and .when can be unsigned, just add one and
divide by two. My thought is that the pooling might be able to be
*replaced* by the multiplication of the elements. The reason for doing
so is that, even though pi is good for selecting sparsity, it's best
logical use is multiplication over the conceptual content itself, just
as sigma does. Finally, your description of this as sigma-pi-not is not
fitting onto negation over symbols, because negation over symbols is
currently pi-sigma-pi-not, the first pi being the basic percept level.
So if we keep that, we should pi-not-sigma, but that does not give us
the end pi to reason over. Can you comment on system 1 and animal
cognition to help us determine?"

This section corrects §19.16 in two places: the stack is named by its
order, Pi, Not, Sigma; and Not sits between a Pi and its Sigma, over
percepts at order zero and over symbols above, not between a Sigma and
the next Pi.

**1. Place and time unsigned.** Where a product reads `.where` or
`.when`, the coordinate is (x + 1)/2 (Alec). §19.16's objection on
signed coordinates is withdrawn.

**2. The pooling replaced by a multiplication of the elements.**
*Checked as a toy* (scratch, `pooling_replace.py`; not the model). Codes
obey rule 5: 12 letters with sparse codes, 40 words of three letters,
a word's code the greatest over its letters plus a keyed increment;
place and time unsigned; width 104; a dense Sigma fitted by least
squares. "Product" multiplies the active percepts' vectors element by
element. "On the complements" is the same multiplication applied to
what each percept lacks: $x_i = 1 - \prod_p (1 - v_{p,i})^{b_p}$.

| Active percepts | Asked | Sum (rule 7) | Product | On the complements |
| --- | --- | ---: | ---: | ---: |
| the word alone | which word, at a new place | 1.000 | 1.000 | 1.000 |
| the word alone | has a letter, of unseen words | 1.000 | 1.000 | 1.000 |
| the word alone | two words read together: which are present | 1.000 | .875 | 1.000 |
| the word alone | XOR of two words, largest error | .5 | 0 | 0 |
| the word and its letters | which word, at a new place | .988 | .075 | 1.000 |
| the word and its letters | has a letter, of unseen words | 1.000 | .750 | 1.000 |
| the word and its letters | two words read together: which are present | 1.000 | .058 | .995 |
| the word and its letters | XOR of two words, largest error | .5 | .5 | 0 |

- *The product itself* gives what the active percepts share, since a
  zero annihilates. Where a word's parts are active beside it, almost
  nothing is shared and the concepts cannot read the word (.075; .750
  is the rate of always answering no).
- *The product on the complements* keeps everything the sum keeps, and
  it also reads the conjunction of two words with no row for the pair
  (error 0, where the sum gives .5), because a + b - ab holds the
  product ab. Two percepts interact only on the coordinates they share,
  which under rule 5 are the parts they share.
- It is the or of the product logic (§19.13), as the Pi's membership
  product is its and. It is monotone, it crosses no lane, a percept with
  $b_p = 0$ changes nothing, and the pooled vector stays between zero
  and one, which the sum does not.
- *Recommended:* rule 7's sum is replaced by the product on the
  complements, for each lane. *Not checked:* the model's own codes; the
  gradient where an element is already at one; the way back of rule 17
  from such a vector.

**3. Where Not sits.** *Checked as a toy* (scratch, `not_placement.py`;
not the model; monotone Pi and Sigma, no bias; six seeds; in brackets
the Not entries used). "Present only" inputs are present or absent, with
no evidence against; the first two tasks are what animals learn (point
4).

| Task | No Not | Not, Pi, Sigma | Pi, Not, Sigma | Pi, Sigma, Not | Pi, Not, Sigma twice |
| --- | ---: | ---: | ---: | ---: | ---: |
| present only: A alone, not A with X | 0 | 6 (2.0) | 6 (1.0) | 0 | 6 (1.5) |
| present only: A or B, not both | 0 | 6 (2.7) | 6 (1.0) | 0 | 4 (1.5) |
| three values: and | 6 (0) | 6 (0) | 6 (0) | 6 (0) | 6 (.7) |
| three values: or | 6 (0) | 6 (0) | 6 (0) | 6 (0) | 6 (0) |
| three values: not | 0 | 6 (2.0) | 6 (1.7) | 6 | 5 (1.8) |
| three values: nand | 0 | 6 (4.0) | 6 (2.2) | 6 | 6 (2.2) |
| three values: xor | 0 | 6 (4.0) | 0 | 0 | 3 (3.3) |
| the 7 functions of two present-or-absent inputs that are silent when nothing is present | 4 | 7 | 7 | 4 | 7 |

- *With no Not,* neither of the animals' two tasks is learned. So
  System 1 needs a Not, as Alec says of `expect` and `intuit`.
- *Pi, Not, Sigma* learns both with one Not entry each, the fewest, and
  every function of what is present that is silent when nothing is. A
  Not entry there says: this percept, or this conjunction, counts
  against that concept.
- *What it cannot do* is xor over three values: a conjunction of a thing
  with a negated thing, where the evidence against is real. That needs a
  Pi after the Not, which is Alec's "does not give us the end pi to
  reason over". Two blocks have it: the Pi after a Not is the next
  block's.
- *Pi, Sigma, Not* only negates a whole answer and learns neither of the
  animals' tasks.

**4. System 1 and animal cognition** (Claude; the references marked
"checked" were looked up on 2026-11-17, the rest are as `doc/Philosophy.md`
already cites them or from memory).

*What an animal has: three things, none of them a negation of a
content.*

1. *Opposites perceived positively.* Light and dark have a channel each
   from the first synapse of the retina: the ON pathway inverts the
   photoreceptor's signal at one synapse and the OFF pathway keeps it
   (checked). Honey bees place an empty display below one (Howard and
   others 2018, checked). So an against at the level of percepts is the
   for of an opposite, which the two producers of 1.8 §57.6 give. No
   operator is involved.
2. *Omission against an expectation.* `doc/Philosophy.md`, "Absence is
   inferred, never perceived": dopamine neurons dip at an omitted reward
   (Schultz, Dayan and Montague 1997), the two signs of an error are
   carried by two populations (Keller and Mrsic-Flogel 2018), and
   Dharmakirti restricts non-observation to what was suitable to appear.
3. *Learned inhibition.* A cue that signals the omission of an expected
   outcome becomes a conditioned inhibitor (Rescorla 1969, checked), and
   animals learn A or B but not both, where a unit for the compound
   takes inhibitory strength (Pearce 1994, checked). Konorski (1967,
   checked through later reviews) has the inhibitor excite a
   representation of "no outcome": the against lane, driven positively.
   In these models the answer is a sum over units for cues and for
   compounds, some of which count against: a Pi, a sign, and a Sigma.
   Excitation and inhibition are carried by different cells (Dale's
   principle, checked), so each sign is a thing in the circuit, as a Not
   entry is.

*What an animal lacks* is negation as an operator on any content:
denial, a "not P" for a P never met, a "not-red" taken as an object.
Bermudez (2003; checked through reviews) calls what animals have
proto-negation, between contraries such as presence and absence.
Children understand logical "no" and "not" only after the age of two
(Feiman and others 2017, checked). Apes pass exclusion tasks (Call 2004,
checked), but the disjunctive syllogism is shown only from three years
and the animals' success may not be deductive (Mody and Carey 2016,
checked). As with `doc/Philosophy.md`'s "What an animal lacks", the
evidence agrees in outline and is contested in detail.

*What this determines* (Claude's reading, for Alec).

- The block is Pi, Not, Sigma, one of each, at every order. With
  `interpret` lacking, an animal has the first block alone. That block
  learns what animals learn, and it cannot make a negated thing an
  object, which is what they lack.
- Negation over symbols stays where it is: it is the second block's Not,
  after the Pi that MTM's promotions and `interpret` supply. The Pi to
  reason over is that Pi, and thought reasons over its symbols with the
  grammar's own operators.
- An animal's Not entries are learned, from omission. The operator `not`
  of a sentence or a thought can be put on any symbol, which is the
  step from proto-negation to negation.
- `expect`, `intuit` and the marker are stacks of Pi, Not, Sigma. One
  pair learns inhibition; two pairs learn everything in the table.
- With the Not entries left out, each block speaks only of what is
  (§19.15).

*For Alec.* (a) Rule 29 of item 1.7: every block and every stack is Pi,
Not, Sigma, monotone, with no bias. (b) Rule 7's sum replaced by the
product on the complements. *Answered in §19.18:* Alec took Pi, Sigma,
Not, so (a)'s order is withdrawn; (b) is open.

*References added* (to be checked again in the documentation pass).
Bermudez (2003), *Thinking without Words*, Oxford. Call (2004),
*Inferences about the location of food in the great apes*, Journal of
Comparative Psychology 118(2). Feiman, Mody, Sanborn and Carey (2017),
*What do you mean, no? Toddlers' comprehension of logical "no" and
"not"*, Language Learning and Development 13(4). Howard, Avargues-Weber,
Garcia, Greentree and Dyer (2018), *Numerical ordering of zero in honey
bees*, Science 360. Konorski (1967), *Integrative Activity of the Brain*,
Chicago. Mody and Carey (2016), *The emergence of reasoning by the
disjunctive syllogism in early childhood*, Cognition 154. Pearce (1994),
*Similarity and discrimination: a selective review and a connectionist
model*, Psychological Review 101. Rescorla (1969), *Pavlovian
conditioned inhibition*, Psychological Bulletin 72.

### 19.18 Decided: Pi, Sigma, Not is the minimal structure (Alec, 2026-11-17)

*Superseded in its order by §19.19:* the first layer is Pi, Sigma, and
the layers above are Pi, Not, Sigma. Rule 29 is restated there. The
toys of this section stand as a record.

*Alec's words.* "I think we go with pi-sigma-not as the minimal
structure of cognition, and sacrifice reasoning over concepts. That
might fit with dharmakirti's notion that contradiction (both) is not a
valid conceptual object."

*Claude's reading of the order,* for Alec to correct: it is as written.
A Pi makes percepts (above order zero, the rows MTM promotes); a Sigma
makes concepts; Not is applied to a concept. It is "pi-sigma-pi-not"
(§19.17) with its second Pi left out. It is not the order §19.17
recommended (Pi, Not, Sigma), and that recommendation is withdrawn.

29. *The minimal structure is Pi, Sigma, Not.*
    - A row of the Pi reads its own lane. A Sigma's weights are zero or
      more, and it has no bias (*Claude's reading:* carried from §19.15;
      without it Not would not be the only crossing). A concept is still
      one signed value (§18.25).
    - Not is the swap of a concept's two lanes (`FieldLogic.reduce`'s
      `not`), and it is the only crossing of the lanes.
    - No Pi is computed over concepts and their Nots: nothing conjoins a
      concept with a negated concept. This is the reasoning over
      concepts that is given up. *Claude's reading:* it is given up in
      the layers; the grammar's operators still act on object concepts
      in comprehension and in thought (1.8 §68.26).
    - So "both" is not a conceptual object: a concept is one signed
      value, its Not is its swap, and no row holds the two together.
      `doc/Architecture.md` has this already of an episode's entry: "One
      signed value ... approximates the pair because a contradiction is
      not a valid mental object (Dharmakirti) ... the both corner
      belongs to working memory only". Percepts, fields and truth rows
      keep both lanes, as now.

*What it changes in §19.2:* rule 2 (a row reads its own lane), rule 3
(weights zero or more, no bias), rule 4 (the Not follows the Sigma),
rule 18 (the stacks are pairs of Pi, Sigma, Not), and rule 23's "a
negative weight passes to the other lane" (a Not does).

*Checked as toys* (scratch, `psn_stack2.py`, `psn_hand.py`,
`psn_sigma.py`; not the model; six seeds; in brackets the Nots used).
The rows of these toys have no diagonal, so a row with no weight is the
constant one; the cycle's rows have their diagonal (rule 2).

| Task | Pi, Sigma, Not | then a Sigma over the concepts and their Nots | then a Pi over them, and a Sigma |
| --- | ---: | ---: | ---: |
| present only: A alone, not A with X | 0 | 6 (1.0) | 0 |
| present only: A or B, not both | 0 | 6 (1.0) | 0 |
| three values: and | 6 (0) | 6 (0) | 6 |
| three values: or | 6 (0) | 6 (0) | 6 |
| three values: not | 6 (1.0) | not run | 6 |
| three values: nand | 6 (1.0) | 6 (1.0) | 6 |
| three values: xor | 0 | 6 (1.5) | 0 |
| three values: a and not b | 0 | 1 (1.0) | 2 to 4 |
| the 7 functions of two present-or-absent inputs that are silent when nothing is present | 4 | 7 | 4 |

- *The structure alone* learns ands, ors, and the negation of a whole
  answer. It does not learn the animals' two tasks of §19.17.
- *With a Sigma that sums concepts and their Nots,* both are learned, in
  every run, with one Not entry each, and so is every function of what
  is present that is silent when nothing is. No Pi over concepts is
  needed. In the tower that Sigma is the next order's (rule 11); in a
  stack it is the next pair's, where each row starts as its own input
  (*Claude's reading* of rule 21's start).
- *With a Pi over the concepts and their Nots* (the last column), the
  two tasks can be built by hand and are exact, but descent found the
  first in 1 of 6 runs of 8,000 steps and the second in none.
- *What is given up* is what Alec names: a conjunction with a negated
  thing as an object (a and not b: 1 of 6 without a Pi over concepts),
  and "both".

*The stacks today* (read in the source). `expectLayers` is 1, and
`intuitLayers` and `markerLayers` are 2. So as it is configured,
`expect` would be one pair: it could negate an answer and could not
learn an exception (A, but not A with X). *Default, for Alec:*
`expectLayers` 2.

*Certificate* (added to §19.5). 23. *Pi, Sigma, Not.* (a) With no Not
entry, raising an input never lowers a concept, and with no evidence
against at the input no concept is against. (b) A concept's Not is its
swap. (c) A Sigma over concepts and their Nots learns "A alone, not A
with X" from random starts, with one Not entry.

*Open.* The three readings above (the order; the monotone rows and
weights; the grammar's operators left as they are); `expectLayers`; and
§19.17's (b), the pooling.

### 19.19 Decided: Pi, Sigma at the first layer; Pi, Not, Sigma thereafter (Alec, 2026-11-17)

*Superseded in part by §19.20:* the basic level is Pi, Not, Sigma too,
and an animal needs no layer above the first. Rule 29 is restated there.

*Alec's words.* ""What animals lack is negation as an operator on any
content" makes me think that the first layer should be Augustinian for
all creatures: pi-sigma. There is good conceptual motivation; negation
can only apply to scalar symbols, becuase the negation of an entire
vector is not biologically easy. Which gives us pi-sigma for the first
layer and pi-not-sigma thereafter. Can animals solve xor? Presumably
not, given what you said. I'm eager to see a short demo or PI used for
combination instead of pooling, but I'm optimistic about it."

29. *Pi, Sigma at the first layer; Pi, Not, Sigma thereafter* (restated;
    it replaces §19.18's order).
    - *The first layer,* for every creature: a Pi makes the basic
      percepts and a Sigma the concepts of order zero. There is no Not.
      It speaks only of what is.
    - *Thereafter:* a Pi makes symbols, Not is the swap of a symbol's
      pair, and a Sigma makes the concepts. Alec's reason: a symbol is a
      scalar and can be negated; at the first layer the Pi's value
      multiplies a whole vector, and the negation of a vector is not
      biologically easy.
    - Rows read their own lane, weights are zero or more, and there is
      no bias (*Claude's reading,* as in §19.18).
    - A row that required a concept both for and against is never
      active, since a concept is one signed value (§18.25). So "both" is
      still not a conceptual object (§19.18).
    - *The stacks.* `expect`, `intuit` and the marker read concepts,
      which are scalars, so each of their pairs is Pi, Not, Sigma
      (*Claude's reading*). One such pair learns the animals' two tasks
      (§19.17: 6 of 6, one Not entry each), so `expectLayers` stays at 1
      and §19.18's question is withdrawn.

*Checked as toys* (scratch; not the model; six seeds; in brackets the
Not entries used). In `layered_nominated.py` the Pi's rows are supplied
as MTM would nominate them (rule 20): each input's own row and a row for
each pair. Only the Sigma's weights and the Not entries are learned.

| Task | First layer alone | With one Pi, Not, Sigma above | With two |
| --- | ---: | ---: | ---: |
| XOR of two perceived words (true and false, each at two places) | 6 (0) | 6 (0) | 6 (0) |
| present only: A alone, not A with X | 0 | 6 (1.0) | 6 (1.5) |
| present only: A or B, not both | 0 | 6 (1.0) | 6 (2.0) |
| not, of a signed input | 0 | 6 (1.0) | 6 (1.0) |
| the 7 functions of two present-or-absent inputs that are silent when nothing is present | 4 | 7 | 7 |

- *The first layer alone* never answers against where the input holds no
  evidence against (lowest answer 0.000 over 20 random networks), and it
  learns XOR over two perceived words with no Not, given a row for each
  pair.
- *One layer above it* learns both of the animals' tasks, with one Not
  entry each.
- *Rows learned by descent* (`layered.py`), in place of supplied rows,
  do worse as layers are added: with one layer above, "A or B, not both"
  4 of 6 and "A alone, not A with X" 0 of 6; with two, neither. This
  agrees with rule 20: the rows are for MTM to nominate, not for the
  gradient to find.

*Can animals solve XOR?* Yes, where it is over what is present. It is
called negative patterning: respond to A, respond to B, and withhold to
A and B together. It is shown in honey bees (Deisig, Lachnit, Giurfa and
Hellstern 2001, checked), and in rabbits and rats (Whitlow and Wagner
1972; Rudy and Sutherland 1989; checked). It is the standard evidence
for a unit for the compound, which is a row of the Pi. In this structure
it needs the layer above the first, with one Not entry. So an animal has
at least one layer of Pi, Not, Sigma over its own concepts, which
`doc/Philosophy.md` allows ("they can have symbols of their concepts").
XOR over two words that are each perceived needs no Not and no second
layer.

*Pi used for combination* (demo:
`/private/tmp/claude-501/-Users-arogers-github-WikiOracle/60c974d9-e29a-405a-b603-4965d33f3a11/scratchpad/pi_combination_demo.py`,
sent to Alec; it is rule 2's Pi, `exp(W @ log(m))`, over the percepts of
a reading, for each element). On the elements themselves it keeps only
what the percepts share (which word: .15; which two words: .06). On the
gaps, 1 less each element, it keeps what any percept has, shared
elements reinforce, and the result stays between zero and one (which
word: 1.00; which two words: 1.00; XOR of two words with no row for the
pair: error 0, where the sum's is .5). *Claude's note:* the gap is what
is left to certainty, not a negation; nothing in it is ever counted
against, so it fits a first layer with no Not.

*Certificate* 23, restated. (a) At the first layer, raising an input
never lowers a concept, and with no evidence against at the input no
concept is against. (b) Above it, a Not is a symbol's swap. (c) One
layer of Pi, Not, Sigma above the first learns "A alone, not A with X"
and "A or B, not both", each with one Not entry, on supplied rows. (d)
The first layer alone learns XOR of two perceived words, with a row for
the pair.

*Open.* (1) §19.17's (b): the sum replaced by the Pi on the gaps. (2)
Whether Alec's "sacrifice reasoning over concepts" (§19.18) still holds
here, where each Sigma is followed by the next layer's Pi. (3) Whether
the first layer keeps an against lane fed by perceived opposites (1.8
§57.6) or has none (*default:* it keeps what perception gives). (4) The
monotone rows and weights, and the stacks' pairs, which are Claude's
readings.

*References added.* Deisig, Lachnit, Giurfa and Hellstern (2001),
*Configural olfactory learning in honeybees: negative and positive
patterning discrimination*, Learning and Memory 8. Rudy and Sutherland
(1989), *The hippocampal formation is necessary for rats to learn and
remember configural discriminations*, Behavioural Brain Research 34.
Whitlow and Wagner (1972), *Negative patterning in classical
conditioning: summation of response tendencies to isolable and
configural components*, Psychonomic Science 27.

### 19.20 One structure, Pi, Not, Sigma, at every order; the basic level reasons over absence (Alec, 2026-11-17)

*Alec's words.* "Sacrifice over reasoning was probably a typo, I'd need
to see it in context. "in your structure an animal needs at least one
Pi-Not-Sigma layer over its own concepts. Philosophy.md already allows
that: "they can have symbols of their concepts"." This remains a
problem, becuase the distinction between human and animal cognition is
recursion, enabled by symbols. So as soon as we have two orders
recursively, we are at human-like mind. Dolphins and chimps might have a
couple orders, but let's focus on the majority mammalian brain. You said
that animals do not have negation, but perhaps negative patterning can
be done as pi-not-sigma, And in some sense this is not negation; it is
reasoning over absence. If so, we have a nice single structure that can
be used for recursion."

*What is withdrawn.* §19.19's "an animal has at least one layer of Pi,
Not, Sigma over its own concepts". The majority mammalian brain is one
order. What tells a human mind from an animal's is recursion, which
symbols enable (Alec), and not the Not. "Sacrifice reasoning over
concepts" (§19.18) is probably a typo (Alec); nothing is built on it.

*Checked as a toy* (scratch, `basic_level_absence.py`; not the model).
One order, no symbols and no second layer. The Pi's rows are the basic
percepts A, B and X and the compounds AB and AX, which are rows of their
own, as MTM nominates a conjunction that recurs; a compound's code is
the greatest over its parts plus a keyed increment (rule 5). The active
percepts' vectors are combined. The Sigma is dense over the content's 96
elements, with for entries and Not entries, each zero or more, and no
bias. Learning is local, one trial at a time: where the outcome came and
was not expected, the for entries of the present content grow; where it
was expected and did not come, the Not entries grow. The numbers are the
expectation of the outcome after learning.

| Taught | Asked | Sum | Pi on the gaps |
| --- | --- | ---: | ---: |
| A, B: outcome; A and B together: none | A | +.92 | +1.00 |
| | B | +.95 | +1.00 |
| | A and B together | +.01 | .00 |
| | nothing | .00 | .00 |
| A: outcome; A with X: none; B: outcome | A | +1.00 | +1.00 |
| | A with X | .00 | .00 |
| | X alone (never shown) | -1.00 | -.98 |
| | B with X (never shown) | .00 | +.04 |

- *Negative patterning is learned at the basic level alone,* as Pi, Not,
  Sigma. So is conditioned inhibition, and the inhibitor passes the
  test animals pass (Rescorla 1969): X alone is expected absent, and X
  takes away what B predicts, though neither was shown.
- *A Not entry is made only by an absence.* Taught A and B with no
  omission, the layer has no Not entry (0 of 96). With omissions it has
  13 to 25 of 96, each from one element of the content to the concept.
- *With the Not entries left out,* the layer expects the outcome to A,
  to B and most of all to both: it speaks only of what is.
- *No vector is negated.* A Not entry is on one element, a scalar, as a
  Not above is on a symbol, a scalar (Alec, §19.19: "negation can only
  apply to scalar symbols").
- The rule is local and uses no gradient passed back (1.8 §68.30).

29. *One structure: Pi, Not, Sigma, at every order* (restated on Alec's
    "If so"; it replaces §19.19's first layer of Pi, Sigma).
    - *At the basic level* the Pi makes the basic percepts, their
      vectors are combined, and the Sigma is dense over the content's
      elements. A Not entry there is from one element to one concept. It
      is taught only by an omission, the lane that already teaches the
      predictor (`doc/Philosophy.md`, "Expectation as a Gain"); it is
      reasoning over absence, not the negation of a content (*Claude's
      reading* of Alec's words). The against lane at this level records
      that something expected is absent, and with no omission the level
      speaks only of what is.
    - *Above it,* for a mind that has symbols, the same structure
      repeats: a Pi makes symbols over concepts, Not is the swap of a
      symbol's pair, and a Sigma makes the concepts. The operator `not`
      of a sentence or a thought can be put on any symbol, which no
      omission need teach.
    - Rows read their own lane, weights and entries are zero or more,
      and there is no bias (*Claude's reading*).
    - `expect`, `intuit` and the marker are stacks of the same structure
      over concepts (§19.19).

*Certificate* 23, restated. (a) With no omission taught, the basic
level has no Not entry, raising an input never lowers a concept, and no
concept is against. (b) The basic level alone learns "A or B, not both"
and "A alone, not A with X" from what was expected and what came, and
after it X alone is expected absent. (c) The basic level alone learns
XOR of two perceived words with no Not entry, given a row for the pair.
(d) Above the basic level a Not is a symbol's swap.

*Open.* (1) §19.17's (b): the sum replaced by the Pi on the gaps (the
toy learns with either). (2) What teaches the Not entries of the basic
level's own Sigma, as distinct from `expect`'s: here an omission
(*Claude's reading*). (3) The monotone rows and entries, and the stacks'
pairs, which are Claude's readings.

*Alec, then (2026-11-17):* "So animal cognition sees particular events
not happening, but it does not negate ideas (I.e. no conceptual
negation, only perceptual absence)."

This is the principle of rule 29's basic level. A Not entry there is
taught by a particular event that was expected and did not happen, and
nothing at that level negates an idea. Conceptual negation is the
operator `not`, on a symbol, and so belongs to a mind with symbols.
*From the record:* the absence is only of what was expected ("one does
not miss what one was not looking for", `doc/Philosophy.md`); the input
holds no negative fact.

*For the documentation pass* that lands with item 1.7: in
`doc/Philosophy.md`, "Absence is inferred, never perceived" says an
absence is "only concluded in thought". To agree with this principle it
tells apart the omission, which System 1 registers against its
expectation and which an animal has, from the statement of an absence,
which thought concludes.

### 19.21 Item 1.7 as it stands after 2026-11-17: the rules to build, in one place (Claude; a doc pass)

Alec, 2026-11-17: "Make sure it's all in the doc, and 1.7". This section
states every rule of item 1.7 once, in its final form, with its
certificates and its order of building. For the build it replaces §19.1,
§19.2 and §19.5 to §19.7; §19.2 to §19.20 are its record, and where they
differ this section stands. Nothing here is new: each rule names where
it was decided, and Claude's readings and defaults are marked as they
are in the record.

**What it builds.** One structure at every order: a Pi, then Not, then
a Sigma. Percepts, and above order zero symbols, are unsigned rows of a
square, sparse Pi, a product of memberships. Concepts are signed and are
a Sigma over them, each entry for or Not. A row exists only when MTM
nominates it. At order zero the percepts' vectors are combined before
the Sigma. Symbols are split from concepts and have no dimensions. The
old classes are not edited.

**A. The layers and the structure.**

1. *New classes* (§18.24, rule 32). `PiLayer3` and `SigmaLayer3`
   (*default names*). `SigmaLayer`, `PiLayer`, `PiLayer2` and
   `SigmaLayer2` are not edited, and nothing built with them changes
   value.
2. *The Pi* (§18.24, rules 21 and 31). Square and sparse, on unsigned
   values: $z = \exp(W \log \mathrm{clamp}(m, \epsilon, 1) + b)$, with
   `PiLayer2`'s constraints (diagonal at one or more, the rest at zero or
   more, bias at zero or less), so a row requires the members it has
   weight on and never exceeds its own reading; its way back is the exact
   LDU solve. Each percept or symbol is two unsigned rows, one for each
   lane (*Claude's reading* of §18.25). A row has weight only on rows of
   its own lane (*Claude's reading* of rule 29).
3. *The Sigma* (§18.24, rule 23; §18.25; rule 29). For each concept one
   signed number, $\tanh(x\,U)$, in $[-1, 1]$, with no bias (*Claude's
   reading* of rule 29). Each input has one signed weight. Its positive
   part is the input's *for entry* and its negative part is its *Not
   entry*, so an entry is never both (*Claude's reading:* the arithmetic
   of §18.24 is unchanged, and the negation is named and can be counted
   and left out; Alec, 2026-11-17: "Signed sigma; good, this is knowing
   in the signed hyperball"). A weight starts at zero or above (rules 8b and 21), so
   there is no Not entry until teaching makes one. The concept's value is
   stored as two unsigned lanes, its positive part on the for lane and
   its negative part on the against lane; a single reading is never
   "both". Where an answer is read from a Sigma, the signed number is
   read (*default*).
4. *The cycle* (§18.25, as amended by rules 20, 21 and 26). At an order
   with $n$ concepts: $n$ signed values, stored as $2n$ unsigned lanes;
   the rows above that MTM has nominated read those lanes, each row its
   own lane; each symbol's pair is combined, for less against, into one
   signed input; the next Sigma reads those inputs.
29. *One structure: Pi, Not, Sigma, at every order* (Alec, §19.20).
    - *At the basic level* the Pi makes the basic percepts, their
      vectors are combined (rule 7), and the Sigma is dense over the
      content's elements (rule 8). A Not entry there is from one element
      to one concept, a scalar. It records that something expected is
      absent. Alec: "animal cognition sees particular events not
      happening, but it does not negate ideas (I.e. no conceptual
      negation, only perceptual absence)". What teaches it is an
      omission, the concept expected and not come, which is what the
      gradient of a weight that starts at zero or above gives (Alec,
      2026-11-17: "Omission teaches not: yes, for percepts, the opposite
      of 1 is 0, not -1, as in Boole").
    - *Above it,* for a mind that has symbols, the structure repeats: a
      Pi makes symbols over concepts, a Not entry is from one symbol to
      one concept (the concept reads that symbol's pair swapped), and a
      Sigma makes the concepts. What tells a human mind from an animal's
      is this recursion, which symbols enable, and not the Not (Alec).
      The majority mammalian brain is one order.
    - The operator `not` of a sentence or a thought swaps the lanes of
      what it is put on (`FieldLogic.reduce`), as now. It can be put on
      any symbol, and no omission need teach it.
    - With every weight held at zero or above, so that no Not entry is
      read, a mind speaks only of what is (§19.15, Shamatha Speech).

**B. Order zero.**

5. *A percept's code obeys the mereology* (§18.24, rule 27): at or above
   each part's code and at or below each whole's. A new row's code is the
   greatest over its parts' codes and an increment (*default:* a small
   draw keyed to the row), in place of the sum of its bytes' codes
   (`BytesFallbackEncoder.encode`).
6. *A basic percept is a row of the Pi* over its parts and wholes
   (§18.5, rule 8, with rule 31), in place of the least of
   `BasicPercepts.read`. For order zero, `store.conjunctive` and the key
   `conceptualPi` go. It becomes a row when MTM nominates it (rule 20).
7. *Combination within one reading* (§18.24, rule 22; §19.17). For each
   lane, the vectors of the active percepts are combined into one vector
   as wide as the perceptual space, and the Sigma reads the for vector
   less the against vector. `.where` and `.when` are read unsigned,
   $(x + 1)/2$ (Alec). Since item 1.8 they are stored so (1.8 §59.62;
   Alec, 2026-11-18, as Codex records it: "Use sinusoids in 0-1
   range, that's a todo in 1.7 anyway."), and a reader of an angle
   takes $2x - 1$; this part of the rule is built. The combination is
   the Pi on the gaps (§19.17,
   §19.19): for each dimension, a Pi over the percepts of the reading,
   taken on one less each value, $x_i = 1 - \prod_p (1 - v_{p,i})^{b_p}$,
   with $b_p$ the Pi's value for percept $p$ on that lane and $v_p$ its
   full vector. It replaces §18.24's sum, $x = \sum_p b_p\,v_p$ (Alec,
   2026-11-17: ""Pi on the gaps". You mean pi over the individual percept
   dims instead of pooling? Good.").
8. *The concepts of order zero* (§18.24, rule 23): the Sigma of rule 3 on
   $x$, with $U$ dense, of the perceptual width by the concepts. It
   replaces §18.5's rule 9, the fold of `ConceptualSpace._compose_order0`,
   the readout `IndexedSigmaConceptsFromPercepts`, and item 1.8's
   stand-in for one signed weight (`SparseLayer._fold_weights`, §58 of
   the 1.8 specification) at order zero.

8b. *A concept made at order zero starts from its members* (*default*):
   a new column of $U$, the sum of its members' vectors, normalized,
   which the network then learns; under rule 21 a new concept has one
   member, its own row. What promotion, `PerceptCollections.collect` and
   the lessons write today (an edge in the feature matrix,
   `add_concept_feature`) becomes such a column. The Pi of order zero
   holds only the mereology's entries, a percept over its parts and
   wholes; no concept writes into it, and it stays full rank. The
   reduced rank of order zero is the Sigma's, at most the perceptual
   width.

**C. Above order zero.**

9. *Symbols are split from concepts and have no dimensions* (§18.22,
   rules 14, 15 and 20). Separate tensors; a symbol has no vector, place,
   form or trust content read as a direction; the shared row of a concept
   and its symbol is given up.
10. *A symbol is two unsigned rows, one for each lane, and brings one
    concept* (§18.25; rules 2 and 21). It is made when MTM nominates it
    (rule 20), or by `interpret` for a meta concept, and not at a
    concept's allocation. A conjunction is promoted as a new symbol, a Pi
    row over its members, under a new concept, a Sigma over that symbol.
11. *The Sigma above order zero* (§18.24, rule 24, with §18.25): it reads
    each symbol's pair combined into one signed input. No vector and no
    pooling. Stored sparse (*default*).
12. *Sigma and Pi at every order* (§18.24, rule 25). The least and the
    greatest of `SparseLayer.fold_presence` go above order zero too, and
    item 1.8's `SparseLayer._fold_weights` with them.
13. *Place and time* (§18.24, rule 26). A percept's vector carries
    `.where` and `.when`; a symbol stores neither; its `.where` is written
    when STM reads it. Where a product reads them they are taken
    unsigned (rule 7).
14. *Between conceptual orders, only activation* (§18.22, rule 16): a
    pull within one order does not pass through a symbol.

**D. Learning and the way back.**

15. *Membership is learned under a penalty* (§18.24, rule 29; rule 22).
    *Default:* §18.3's penalty, measured first, on the Pi's entries and
    on the Sigma's entries above order zero. The Sigma of order zero is
    dense and is not penalized. Above order zero a promotion writes an
    edge at one into capacity allocated beforehand; at order zero it
    writes a column of $U$ (rule 8b). The penalty thins the values of
    entries that exist; it does not make a matrix sparse (§18.9).
16. *Likeness of codes pairs nothing* (§18.24, rule 28, *default*). A row
    enters a set, or two rows are taken as neighbours, by the mereology or
    by a promotion, never by the likeness of their codes. The readers that
    decide by likeness today are §19.4, list 1.
17. *The way back* (§18.24, rule 30). The Pi inverts exactly; the Sigma
    returns the combined vector by least squares where there are enough
    concepts, and not otherwise. A signed value on the way back is
    replaced by its two unsigned parts before a percept or a symbol
    holds it, as on the way up (Alec, §19.24): -.5 is .5 on the
    against lane and nothing on the for lane. Teaching does not assume
    an exact inverse; where it does today is §19.4, list 2.

**E. The learned maps of System 1, and storage.**

18. *The stacks are the same structure* (Alec, §19.10; rule 29).
    `SigmaPiLayer` and `SigmaPiStack` (`expect`, the read from a field to
    a point, and the marker of `bin/Language.py`; `intuit`'s own stack
    goes with item 1.8, §19.22) are pairs of rules 2 and 3, a Pi and then
    a Sigma
    whose entries are for or Not, with no way back, fused so that a stack
    is captured as one graph. They read concepts, which are scalars
    (*Claude's reading*, §19.19). Today `bin/SigmaPiStack.py` uses
    `PiLayer`'s chart of odds and a linear `SigmaLayer` clipped to [0, 1].
    This replaces the formula of §18.5, rule 5; rules 6 and 7 of §18.5
    (keys, pair count, seed, the last Sigma starting at zero, the cost of
    an MLP layer) stand. `expectLayers` and `ideaLayers` stay at 1, and
    `markerLayers` at 2.
19. *The matrices are tensors on the device* (*default*): the Pi and the
    Sigma above order zero are stored as compressed rows, not as
    dictionaries of edges (`SparseLayer._index`), and grow into capacity
    allocated beforehand. They can be read by row and by column (rule 28;
    *default:* an index of columns kept beside the rows).

**F. Rows and concepts** (§19.10).

20. *MTM nominates every row of the Pi* (Alec). A basic percept at order
    zero, and a symbol at every order above it, becomes a row only when
    MTM nominates it on recurrence (`concept_mint_count`); nothing
    becomes a row on its first reading. *Claude's reading:* the tower of
    parts and wholes below the basic percepts is perception and is made
    on first sight, as now. This replaces the basic percept reserved at a
    word's first reading (`BasicPercepts.reserve` in the word loop of
    `bin/Models.py`), the lessons' declared admission
    (`Corpus.admit_training_forms`, `ConceptLessons.admit_native_forms`),
    and symbols made at a concept's allocation.
    - *A new word* is perceived at once, but `interpret` cannot find it
      until its basic percept is nominated; its concept comes with that
      row (Alec, "A,b,c,d, yes").
    - *Not nominated* (Alec): the meta concept of a word and its object,
      which `interpret` makes, with its structure, when it interprets the
      word and the meta concept does not yet exist; the blank object,
      below; and the grammar's declared concepts, the 72 arguments and 27
      operator classes of §19.9, each declared with its row at
      construction.
    - *The 12 frozen thought predicates* of §19.9 stay columns of $U$ at
      zero, frozen, read by thought as references and never by
      perception (Alec, "E nice!").
    - *The blank object* (Alec). For a word with no object, `interpret`
      makes a new concept with no members, together with the meta
      concept; today MTM holds such a word as a pending name
      (`InterpretLayer.reading`, `pool.named`). *Default:* it is one
      order above its word. Its vector starts at zero (Alec: "Pure zero
      is even better, since magnitude is certainty in that space"); a
      read that normalizes a concept vector must not divide by its zero
      length.
21. *One concept for each row* (Alec). Each nominated row brings one
    concept, so every order's Sigma is square: as many concepts as
    inputs, with no halving between orders. A concept starts reading only
    its own row, at one; at order zero that is the column of $U$ given by
    the row's own vector, normalized. Concepts have no nomination or
    promotion of their own: MTM's promotions of classes, identities and
    percept collections become nominations of rows. The one writer of a
    concept's further inputs stays: a kept reading adds the word that
    filled an argument to that argument
    (`ConceptHierarchy.observe_member`), as an entry on the word's symbol
    once MTM has nominated it.
22. *Both thinned* (Alec). Rule 15's penalty is on the Sigma's entries
    above order zero as well as on the Pi's; §19.13 shows that with the
    Pi alone penalized what is learned cannot be read from the rows. The
    structure of a meta concept is never thinned: neither penalty acts on
    it. *Claude's reading:* a definition between concepts ("balloons are
    rubber", an edge taught from its `LTM1` part row by `DefinitionFold`)
    stays under the penalty, held by its witness's cost.

**G. The vertical spread** (§19.11).

23. *A fan up and a share down,* on the structure's own matrices, Pi and
    Sigma alike, by the size of their weights. Up: a concept's or a
    symbol's value is split among the rows that read it, each by its
    weight's share of all the weights it is read with (its fan). Down: a
    row's value is split among its inputs, each by its weight's share of
    the row. What a row receives is summed over all its parents and
    children, at most one; a Not entry passes to the other lane. At
    order zero the way down goes on into the first layer (rule 30); the
    combined vector of order zero's Sigma is not passed (rule 17).
24. *Siblings get less than the concept that caused them* (Alec). A
    concept present at one gives every other concept the spread reaches
    less than one: what goes up is split by its fan, and of each parent's
    value the concept itself takes back a share.
25. *One pass, written to the expected field at every order, where the
    grammar calls for it* (as amended by Alec, §19.22). Up once, then
    down once, from order zero's expectation (`expect`) and the observed
    field. It writes `_expected_knowing` at every order, where
    `ConceptSpread.open_prediction` writes order zero alone today. It is
    what `intuit` runs after it deposits expect's field; expect itself
    is the map at order zero alone, and a model with no symbols never
    runs the spread over the orders. *Default:* it also runs at the end
    of the read, before comprehension takes its leaves' objects (1.8
    §68.26), so that `InterpretLayer.preferred_objects` reads what the
    sentence's words primed.
26. *Not strictly ramified* (Alec). A symbol of order $k$ reads the
    concepts of every order below $k$: "the field of a second order
    symbol is both zero and first order concepts". A meta concept is one
    order above the higher of its word and its object and reaches both in
    one step, with no symbol between. *Claude's reading:* it holds for
    every symbol, not only meta concepts.
27. *Nothing new to learn.* The spread reads the structure's matrices,
    adds no parameter and no teacher, writes only the expected field, and
    changes no value of the evidence.
30. *The spread reaches the first layer, and the way back reads what it
    expects* (Alec's request of §19.23; the form is Claude's). The fan
    and the share run on the Pi of order zero, the mereology's own
    entries: a part that is present raises the rows that read it, and a
    row gives a share to each of its parts, so a part that is missing
    receives from the rows its neighbours raised. It writes expected
    percepts and changes no evidence; a row still requires its members
    (rule 2). At the reconstruction, where a look's percepts were
    withheld the way back reads the expected ones in their place
    (*default*). A concept reaches its row directly (rule 21).

**H. Thought reads the structure** (§19.12).

28. *Thought reads the Sigma and the Pi* (Alec). `parts(c)` gives the
    entries of c's row of the Pi: the symbols below c, at every lower
    order (rule 26), each with its weight. `wholes(c)` gives the rows of
    the Pi above c that have c as an entry, each with its weight. Of the
    Sigma, thought can read what a concept reads (its own row, and the
    inputs `observe_member` adds), which concepts read a given row, and
    which of those entries are Not entries. A result is a set of
    references with their weights, not a yes or no. A reference's words
    are those whose meta concepts join them to its concept; a reference
    with no word, such as a percept, is given as itself. An entry is read
    as present by a low threshold, since an entry that survives the
    penalty is small (§19.13; *default:* the threshold is a key, measured
    with rule 15's penalty). *Defaults:* a read is serial and is charged
    to the work budget as the queries are today; it moves nothing.

**I. The order of learning** (§19.25).

31. *Words, objects, then their meta concepts* (Alec). A word
    generalization and an object generalization are each learned
    from their own occurrences, each a row that MTM nominates with
    its concept; the object's may be of another modality. An
    occurrence keeps its place and time, and a generalization holds
    neither. Then the meta concept of the two is learned from their
    consistent association. Nothing joins independent tokens: "there
    isnt a statistic that will magically produce identity across
    independent tokens". Two words are joined through what both are
    joined to. The clauses in this item's terms, and how it meets
    rule 20's meta concept and blank object, are in §19.25.

**J. Looks and spellings** (§19.28).

32. *A look is a `.where` that returns percept indices, and a Pi finds
    the spellings* (Alec, 2026-11-18). A look is not made of the stem's
    word intervals: "If that is forced to words, then the lexer is
    doing the models job". Different spellings of a word ("caat",
    "cat") share most of their parts and wholes, so a row of the Pi in
    that neighbourhood finds them and returns the basic percept that
    corresponds to all of them. The hit or miss on a word feeds the
    width of that row's cone. What is to be specified is in §19.28.
    The contract of a look, and what items 1.8 and 1.5 build of it,
    is 1.8 §74: the look's read is the mask and then the Pi, with no
    reference to word intervals, and a look with the same mask gives
    the same result whatever was looked at before it. Alec's note on
    it: the mereology is built once for the sentence, and the Pi
    creates the basic percepts "as something that changes in
    dependence on the .where supplied by the attend() mechanism".

**Certificates.**

1. *The old classes are unchanged:* `SigmaLayer`, `PiLayer`, `PiLayer2`
   and `SigmaLayer2` give the same values before and after, on random
   inputs over the whole range.
2. *The Pi:* §18.24's certificate 21 (a row requires its members);
   bounds over the whole range, rails included, with weights scaled far
   beyond their start (§18.6, certificate 4); the exact way back (atol
   1e-6, §18.6, certificate 5); a row has no weight outside its own lane.
3. *The Sigma:* each concept is one signed number in a reading and its
   two lanes are its parts, never both above zero (Alec's numbers,
   §19.24: .5 is for .5 and against 0; -.5 is for 0 and against .5);
   on the way back a value of -.5 reaches the percepts as .5 on the
   against lane and nothing on the for lane; no symbol and no percept
   holds a number below zero; bounded as in 2; it has no bias; an
   entry is for or Not, never both.
4. *The cycle:* the next Sigma reads one combined input, for less
   against, for each symbol; no symbol is made at a concept's allocation;
   a promotion is the nomination of a row, which brings its one concept.
5. *Order zero:* §18.24's certificates 15 (a new percept needs no new
   weight), 17 (concepts told apart, reported), 18 (the mereological
   order), 19 (the way back as far as it goes) and 20 (place and time
   weights, reported). Its certificate 14 (pooled equals flattened) goes
   with the sum: the combined vector is one less the product of the
   gaps, lies between zero and one, and a percept at zero changes
   nothing.
6. *Symbols:* §18.22's certificates 11 and 12 (no tensor holds both a
   concept and a symbol; nothing reads a vector, place, form or trust
   content at a symbol; a Sigma above order zero is unchanged when the
   places of every other order are rotated).
7. *The stacks:* a fused `SigmaPiLayer` equals its two layers (atol
   1e-6); a stack is one graph; seconds against an MLP layer of the same
   size; depth (§18.6, certificates 3, 6, 7 and 8, on rule 18's
   structure).
8. *Likeness and the way back:* no reader of §19.4's lists 1 and 2, or
   of the table below, is left as it was, each changed as listed or
   reported.
9. *The standing gates and the lesson* (§18.24, certificate 16; §18.6,
   certificate 10): both stopping gates pass, XOR among them; the lesson's
   readings before and after, each order's occupancy, the Pi's kept edges
   for each row, the Not entries at each order, and one default sweep.
10. *Speed* (*default*): the seconds of the lesson's sixteen sentences
    before and after, as §51 of the 1.8 specification reported them.
11. *No row on a first reading.* A new word read once has no basic
    percept and no concept, and `interpret` does not find it; after
    `concept_mint_count` readings MTM nominates its basic percept, its
    concept comes with it, and `interpret` finds it.
12. *Square.* At every order the concepts equal the rows in number, and
    a new concept reads only its own row, at one; at order zero its
    column of $U$ is its row's vector, normalized.
13. *Thinned.* Under the penalty, a Sigma entry above order zero that no
    cost supports goes to zero, as a Pi entry does; the Sigma of order
    zero is not penalized.
14. *Up.* A word found raises its meta concept by its share.
15. *Down.* A primed meta concept raises its object when the object is
    absent; the way back of rule 17, on the same case, does not.
16. *Rule 3 of 1.8 §55.1.* Finding a word raises its meta concepts and
    their objects; of a word's two objects with equal expectation, the
    one whose meta concept another word of the sentence primed is
    preferred at comprehension.
17. *Evidence unchanged.* With the spread on and off, every order's
    evidence is equal, and the parameters are the same.
18. *The blank object.* Interpreting a word with no object makes one
    blank object and one meta concept, and the blank object's vector is
    zero; interpreting it again makes nothing new.
19. *Siblings below the cause.* With one concept present at one, every
    other concept the spread reaches is below one; two children sharing
    five parents of equal weight give each other .5.
20. *One step across orders.* A symbol of order two reads concepts of
    orders zero and one; a meta concept over a word of order zero and an
    object of order one primes the object in one pass, with no symbol
    between.
21. *The structure is readable.* For a concept c, `parts(c)` and
    `wholes(c)` give the references and weights of c's row of the Pi and
    of the rows above that have c as an entry, with the words of those
    that have words, and the Not entries are told apart; the matrices,
    the activations and the rows are unchanged after the read.
22. *And, or and not.* A stack of rule 18 learns and, or and not over
    true, neither and false from random starts; and and or use no Not
    entry; with rule 22's penalty, the network whose present entries are
    set to one and the rest to zero gives the same answers.
23. *Pi, Not, Sigma.* (a) With no omission taught, the basic level has
    no Not entry, raising an input never lowers a concept, and no concept
    is against. (b) The basic level alone learns "A or B, not both" and
    "A alone, not A with X" from what was expected and what came, and
    after it X alone is expected absent. (c) The basic level alone learns
    XOR of two perceived words with no Not entry, given a row for the
    pair. (d) Above the basic level a Not entry is a symbol's pair read
    swapped.
24. *A blind spot is filled* (Alec, §19.23; the testpoint
    `TestBlindSpot`). With half of a known word's forms withheld, the
    reconstruction gives the whole word at its place; the word beside it
    is unchanged; a word never taught is not completed. The values of
    §19.23 are printed.
25. *A word and an object of another modality are tied by their
    consistent association* (Alec, §19.25; the testpoint
    `TestCrossModalTie`). The word alone and the stimulus alone each
    make a row and a concept, and no tie; presented together they
    make the meta concept; then each leaves the other expected; a
    word met with a different stimulus each time is tied to none; a
    second word met with the same stimulus is tied to the same
    object. The values of §19.25 are printed.
26. *A sequence is learned* (Alec, §19.25; the testpoint
    `TestBellSalivate`). After the first stimulus has been followed
    by the second, the first alone leaves the second expected, and
    its response through it; a stimulus never followed by the second
    does not. The values of §19.25 are printed.

**The order of building.**

1. The two classes (rules 1 to 3 and 29), with certificates 1 to 3.
2. Order zero: codes (rule 5), the percept Pi (rule 6), nomination and
   one concept for each row (rules 20 and 21), combination and the Sigma
   (rules 7, 8 and 8b), with certificates 5, 11, 12 and 23 (a) to (c).
3. Symbols and the orders above (rules 4, 9 to 14 and 22, the meta
   concept and the blank object), with certificates 4, 6, 13, 18 and
   23 (d).
4. The vertical spread (rules 23 to 27, and rule 30 of §19.23), with
   certificates 14 to 17, 19, 20 and 24. Then rule 31, which needs
   steps 2 to 4, with certificates 25 and 26.
5. Thought reads the structure (rule 28), with certificate 21.
6. The stacks (rule 18), with certificates 7 and 22.
7. The readers of §19.4 and of the table below (rules 16 and 17), with
   certificate 8.
8. Certificates 9 and 10.

**Readers to adapt.** §19.4's four lists stand. Added on 2026-11-17 (read
in the source; Codex takes them again in step 1 of §19.8):

| Where | Today | Under |
| --- | --- | --- |
| `ConceptAllocator.class_member` | admits a membership only between adjacent orders | rule 26: any lower order |
| `ConceptSpread.open_prediction` | writes `_expected_knowing` at order zero alone | rule 25: every order |
| the thought queries `parts` and `wholes` (`bin/Queries.py`, `_neighbors`), through `Taxonomy.capture_taxonomy` | read LTM1's part table (`allocator.parts`) | rule 28; which of the two they read is open |
| `ConceptualSpace.concept_parts` and `concept_wholes` (`bin/Spaces.py`) | read the allocator's part and whole references of a concept | rule 28's reads of the Pi, where the matrices replace those references |
| `InterpretLayer.reading`'s pending name (`pool.named`), and `PrependedSymbols.stage`, which holds it in STM | a word with no object is a pending name in MTM | the blank object (rule 20); item 1.8's comprehension takes the pending name as a leaf's object and reads the concepts held in STM (1.8 §68.19, §68.25), and follows it |
| `SigmaPiStack` (`bin/SigmaPiStack.py`) and its keys `expectLayers`, `markerLayers` and, once item 1.8 has replaced `intuitLayers`, `ideaLayers` | `PiLayer`'s chart of odds and a linear `SigmaLayer`, clipped | rule 18 |
| the word step's payload in `bin/Models.py` (`activation.abs()`, two places), `Interpret.positive_evidence`, `ModelAttention.reference_evidence`, and the walk's own pair (`ModelAttention.native_word_poles`, `Models._attention_sentence_payload`) | a word has two reads: the walk's pair fills the two lanes wherever the walk ran, and the readout's absolute value is the presence of the form; its sign is dropped, and where the walk gives no pair the pair is (presence, 0) | rule 3 and §19.24: one Sigma, whose positive part is the for lane and whose negative part is the against lane; the walk's separate pair goes with the fold (rules 8 and 12). Measured in item 1.8 (1.8 §68.63): the walk's pair was on 373 of 373 deposits of certificate 81 and at least 78 of 97 of the math chain and the gates; none of 478 reads was below zero; with the pair taken from the read's sign where the walk gives none and every read negated, two of four runs failed in MTM (an identity whose only mention is marked as an alternative has no member in the actual timeline, and `PrependedSymbols.symbolize` raises), so the readers of a word stored against are traced first. Item 1.8's certificate (with every read negated, no stored number is below zero) stays; where it reports equal values, this item changes that to the lanes exchanged |
| `ConceptualSpace.promote_symbol_reference` and `compute_symbolic_reference` (`bin/Language.py`), in `stage_cs_sym` | a symbol carries a row and one signed activation ("SymbolSpace transports identity and signed activation") | rule 10 and §19.24: two unsigned rows, one for each lane (added 2026-11-18) |
| `MediumTermMemory.match_candidate`, `FamiliarTier.witnessed_match` and `FamiliarTier.observe_field` | a candidate is the same as a held one by a cosine of .8 or more (`conceptMatchCos`), after an exact test of proportions, and is in use by a cosine above .5 (`conceptUseFloor`) | rule 16, with parthood both ways in place of the cosine (`doc/Mereology.md`; 1.8 §70.15 and §70.16, Alec, 2026-11-18: "agree"). A cosine ignores size, and size is certainty, so the bar is set by a trial (added 2026-11-18) |

**Defaults and readings, for Alec to overrule.** The names `PiLayer3`
and `SigmaLayer3`; each percept or symbol as two unsigned rows; a row
reading its own lane only; the Sigma with no bias; the answer of a Sigma
read as its signed number; the keyed increment of a new row's code;
sparse storage above order zero; the penalty of §18.3; likeness pairs nothing (rule 16); the
matrices as device tensors with an index of columns (rule 19); the
stacks reading concepts as pairs of the same structure (rule 18); the
blank object one order above its word; the spread run at the open pass
and before comprehension (rule 25); the threshold key and the work
budget of rule 28; the speed certificate.

**Open, none blocking the build.**

1. Whether the queries `parts` and `wholes` read the matrices alone,
   with LTM1's part table kept as the evidence for `isPart`'s proofs
   (*recommended*), or both (§19.12; Alec: "We can spec it later").
2. On graded readings the layers give the product logic and the
   grammar's field operators the least and the greatest (§19.13);
   *default:* they are left to differ.
3. Whether `data/shamatha.grammar` keeps its `not` rule (§19.15).
4. Whether the basic level's against lane is fed by perceived opposites
   (1.8 §57.6) as well as by omissions; *default:* it keeps what
   perception gives.
5. What becomes of the attention walk's own pair once the one Sigma's
   value is the pair (§19.24); *default:* it goes with the fold that
   reads it.
6. How rule 31 meets rule 20. *Alec, 2026-11-18 (§19.27):* better
   never to have the blank; to avoid a discontinuous decision both
   may be trained, and the blank removed only if the two vectors come
   to overlap in conceptual space. Still open: what removing is,
   beside concepts kept forever, and what overlap is (*default:* the
   match of 1.8 §70).
7. What holds a succession, the bell and then the food, so that the
   first leaves the second expected across two sentences. *Default:*
   a row that MTM nominates for the two when the succession recurs,
   read by rule 30's spread, with expect's stack beside it as §19.22
   proposes. Within one sentence expect does it today (§19.25).
8. Rule 32's unspecified parts (§19.28): what a look's `.where` is
   when it is not word intervals; which quantity is a cone's width and
   how a hit or a miss moves it; what a hit and a miss are; whether a
   letter's place counts.
9. "There is a word there" as inclusion in a meta concept (§19.29):
   which row is tested, what a word with no meta concept is in a
   tree, and how `interpretBias` is lowered by experience.
10. `interpret` takes the largest word that fits the look (§19.30):
    what largest measures, overlapping large words, and what makes
    several words one word.
11. A look carries a place and, optionally, a primed whole (§19.31):
    who sets the condition, what a graded priming does, and which
    row it names. As wires (§19.32): a mask on what is active, asked
    for from above; which wires take its gradient, and what a
    returned token is. *Consolidated, 2026-11-18:* points 8 to 11 and
    §19.31 to §19.45 are stated as tentative rules 33 to 41, with
    seven questions for Alec, in §19.46.

**For the documentation pass** that lands with item 1.7 (`doc/*.md` is
not edited now: another pass over those files is uncommitted, in the
worktree `basicmodel-docs`).

| File | What it gains | From |
| --- | --- | --- |
| `doc/Architecture.md` | the one structure, Pi, Not, Sigma, at every order; MTM nominates rows, one concept for each row, both thinned; the vertical spread; a section "Biological plausibility": the two lanes and the Not entries beside the retina's two channels and inhibition carried by its own cells, the local rule of excess and omission, and what is not plausible, inverses and backpropagation | rules 20 to 29; §19.17, §19.20; 1.8 §68.30 |
| `doc/Philosophy.md` | Shamatha Speech as speech with the Not entries left out, with Augustine and the three natures; perceptual absence and no conceptual negation, with "Absence is inferred, never perceived" telling the omission from the statement of an absence; "What an animal lacks": recursion, which symbols enable, and one order for the majority mammalian brain; negative patterning and conditioned inhibition as reasoning over absence; the references of §19.17 and §19.19, checked again | §19.15, §19.17, §19.19, §19.20 |
| `doc/Orders.md` | not strictly ramified; the meta concept one order above the higher of its word and object; the blank object and its zero vector | rules 20 and 26 |
| `doc/Logic.md` | and, or and not in the layers; the product logic on graded readings beside the grammar's least and greatest | §19.13, §19.15 |
| `doc/Reasoning.md`, `doc/QueryContracts.md` | `parts` and `wholes` read the structure, Not entries told apart | rule 28 |
| `doc/Mereology.md`, `doc/Spaces.md` | a percept's code obeys the mereology; combination by the Pi on the gaps; place and time read unsigned | rules 5, 7 and 13 |
| `doc/Params.md` | the keys that go (`conceptualPi`, `symbolTower`) and those added (the threshold of rule 28) | rules 6, 9 and 28 |
| `doc/Language.md`, `doc/Training.md`, `doc/Philosophy.md` | the order of learning (words, objects, then their meta concepts by consistent association) with the example of the well-house, as written in §19.25, in `Language.md` where the meta concept is described and in `Training.md` where the curriculum's order is given; Percy's signal and name, and the references, in `Philosophy.md` (Alec, 2026-11-18: "Please make sure this example of meta-concept learning is in the doc.") | rule 31, §19.25 |

*Confirmed by Alec, 2026-11-17, of this section's three readings.*
"Signed sigma; good, this is knowing in the signed hyperball." ""Pi on
the gaps". You mean pi over the individual percept dims instead of
pooling? Good." "Omission teaches not: yes, for percepts, the opposite
of 1 is 0, not -1, as in Boole". Rules 3, 7 and 29 above carry them, and
they are no longer among the defaults. So a percept lies between zero
and one and its opposite is zero, and a concept lies in the signed
hyperball. *Claude's note* (Boole's algebra, cited from memory): on
percepts a row of the Pi is its and, $xy$; a gap is its not, $1 - x$;
and the Pi on the gaps is its or, $1 - (1 - x)(1 - y)$.

*Alec, 2026-11-17, of MTM's form.* "The form of MTM can probably be the
same as percepts or symbols, since it serves as a testing ground for
percepts or symbols. So an easily promoted and easily forgotten store,
as opposed to the percepts that they promote into." It bears on rule 20
and is taken up in the wiring part of item 1.5, which was item 1.6
until 2026-11-18
(`doc/plans/2026-11-17-item1-6-zero-data-movement.md`, step B); item
1.7 builds rule 20 on MTM as it is.

### 19.22 intuit runs the vertical spread; one mechanism for the two large maps (Alec, 2026-11-17; Claude's answer)

Alec, 2026-11-17 (in full in 1.8 §68.57): "intuit is only called from
grammar, and we have an opportunity for higher-order cognition there,
so lets run the N:N intuit() as the existing propagation of N:N
expect() *as well as* the (up and down) vertical order propagation that
is enabled only with symbols and recursive cognition." And: "So
expectation is N:N, and currently it uses a pi-not-sigma that runs from
concepts to concepts. That is so close to the pi-not-sigma that runs
from basic percepts to concepts that I wonder if there is a way to
integrate the two, because these two N:N computations are the largest
in the model." And: "one possibility would be to involve the
reconstruction to wind up back in basic percepts, which is a kind of
imagination rooted in the word: pi-not-sigma-sigma^-1-not-pi^-1".

**What it changes in this item.**

- *Rule 25 is amended.* The vertical spread (rules 23 to 27) is what
  `intuit` runs, after it deposits expect's field, when the grammar
  calls it. expect is the map at order zero alone. Rule 25 had the
  spread at the open pass of every sentence, as part of expectation at
  every order; that was a default of Claude's.
- *Rule 18 is amended by item 1.8.* `intuit_stack` and `intuitLayers`
  go there (1.8 §68.53, rule 4), and a stack for the read from a field
  to a point is added (rule 5 there). This item rebuilds the stacks
  that remain.
- *Open for Alec.* Rule 25 also ran the spread at the end of the read,
  so that `InterpretLayer.preferred_objects` reads what the sentence's
  words primed (certificates 14 to 16; rule 3 of 1.8 §55.1). `interpret`
  is the grammar's as `intuit` is. *Claude's default:* the spread runs
  wherever the grammar reads above order zero, at `interpret`'s choice
  of an object and at `intuit`; a model with no symbols never runs it.

**One mechanism for the two large maps (Claude's answer, for Alec).**

The two, as this item has them: the first layer (rules 2, 3, 6 and 7:
percept rows to basic percepts by the Pi, their vectors combined, a
dense Sigma from the combined vector to order zero's concepts), and
expect (rule 18: a stack of its own from order zero's field to order
zero's field).

Three ways toward one, and what is known of each.

a. *The way up and the way back alone,* the round trip Alec names. An
   exact way back returns what went up: an omitted value of .8 came
   back 0 (1.8 §3.19), and "an inverse reconstructs, it does not
   complete" (§19.11). So alone it gives the visible input back and
   expects nothing. As built the same holds by teaching: the reverse of
   the open field is scored on the visible events only
   (`SentenceField.reconstruct`, `Omission.visible_events`), and an
   omitted input is restored through System 2's generate walk
   (`reconstruction_order_zero`).
b. *The round trip from a field that something else has lit.* When the
   concepts the way back starts from were lit by more than their own
   percepts, it gives percepts that were not in the input. That is the
   imagination he names: a word lights its concept, the spread lights
   its meta concept and its object, and the way back gives the object's
   percepts. It needs no new matrix. What lights the concepts is the
   spread, or what `intuit` deposits.
c. *The spread carried into the first layer.* Rule 23 stops going down
   at order zero's concepts. Carried one layer further, a concept's
   value goes down by share to the content it reads and to the percept
   rows that give that content, and up by fan to every concept those
   rows support. Order zero's expectation is then the first layer read
   down and up, and expect has no stack. It knows what the first
   layer's rows and weights hold: likeness through shared content, and
   co-occurrence only where MTM has nominated a row for the things that
   occur together (rule 20).

Taken together, b and c are one mechanism at every level: up by the
structure, a fan and a share at the concepts, and back down by the
structure. In the first layer it needs no symbols, which is expect and
an animal's. Across the orders it needs them, which is intuit. The two
large maps are then one set of matrices read in both directions.

*What it would cost.*

- expect would have no teacher of its own. Today it is taught the
  omitted input directly and can learn two words that merely occur
  together. In the toy of 1.8 §3.18, likeness alone reached 3 of 4
  cases of an expectation that depends on a pair, and a container
  reached 4 of 4. Without a row for the pair, the one mechanism does
  not expect one word from the other.
- Omission would still teach, by rule 29: the Not entries of the first
  layer.
- Rule 17 gives the way back of order zero's Sigma a combined vector
  only. The step from that vector to percept rows is a read by
  coverage, not an exact inverse. It would complete by parts and
  wholes, and it has not been tried.

*Claude's proposal.* Build it in this item as a measured alternative
beside expect's stack, which is kept on hand (Alec, 2026-11-14: "keep
the old form on hand in case we want to revert"). The measures: the
test that expectation restores an omitted word
(`test_item18_expect.py::test_81_expectation_restores_the_omitted_native_word`),
and the parameters and the time for a sentence, for both forms. Decide
by those. Nothing of it is in item 1.8.

*Sizes, measured in the math-chain model:* 1.8 §68.57.

**Alec, of the round trip, later the same day.** "it doesnt do a lot of
prediction, but it does reconstruction of missing data based on
(limited) top-down information, and it provides a way (when integrated
into intuit) to do imagination." And: "It also dovetails with The Whole
Part, where I detail the imagination as the inverse of perception. that
is relevant because a forwards-backwards ping-pong between those two
spaces gives us the ability to imagine, which the recursive symbol look
is absent imagination (since symbols are dimensionless). Sound good?"
(Claude reads "look" as "loop". The book is cited as attached to his
ORCID record, 0000-0002-6570-7571, with no passage.)

*Claude's answer: yes, and his own words carry the one condition.*

- The way back is the right way down, and it is the only one that can
  give anything to look at: symbols have no dimensions, so the loop
  over symbols alone imagines nothing. This agrees with way b above.
- "Limited top-down information" is exact. The round trip restores what
  the concepts hold beyond what the percepts lit, and nothing else: an
  exact way back from the concepts the percepts lit returns those
  percepts (way a above). So it predicts little, as he says, and what it
  reconstructs depends on what lights the concepts from above.
- With intuit, the spread over the orders is what lights them: a word,
  its meta concept, its object, and then the way back to the object's
  percepts. That is imagination rooted in the word, and the back and
  forth between the two spaces is one more pass of the same two ways.
- It makes the two large maps one only if order zero's own top-down
  information also comes from the first layer's matrices (way c above)
  and not from expect's stack. Where it does not, the stack stays as
  the source of that information. This is the point to measure.
- As built, the direct way back from the open field is taught the
  visible input only (way a above). For it to fill missing data it must
  either start from a field that expectation or intuit has added to, or
  be taught the whole input as the generate walk's reconstruction is.

*Decided in direction, Claude's reading of "Sound good?" answered yes;
built in this item with the layers and their ways back. Its form to
settle here:* the round trip as the way down for intuit (imagination)
and for expectation's missing data; whether expect keeps a stack as its
source of top-down information at order zero, by the measure of the
proposal above.

### 19.23 A blind spot is filled on reconstruction: a rule and a testpoint (Alec, 2026-11-17)

*Alec:* "please add a testpoint that verifies the architecture fills in
a blind spot on reconstruction (i.e. the pi should generalize of
neighboring elements to fill in a missing area)."

**What a blind spot is here.** Input that is withheld, by the
architecture's own means (`Omission.Draw`): a whole word, or some of the
forms of a word that is otherwise kept. A word of the small lesson model
has 11 forms, each spanning the word. Withholding a form blanks the
bytes it spans (`Omission.prepare`, `stage`), so with half of a word's
forms withheld the word's bytes are the missing area, and its other
forms are the neighbours that remain to be read.

**What the source does today** (a probe, not the model's tests;
`blindspot/blind_spot_probe.py` in the session's scratchpad; the
committed source, HEAD `5721692530`; the small lesson of
`test_item18_expect.py`, 200 presentations of "alpha gamma" and "beta
delta"; three lessons: taught with everything visible, with the second
word withheld, and with every second form of the second word withheld).
Bytes right at the second word's place, of 5:

| Blind spot at the probe | The way back from the open field | The way back through the generate walk |
| --- | ---: | ---: |
| none | 5 | 0 |
| the whole word | 0 | 0 |
| half of the word's forms | 0 | 0 |

The three lessons give the same numbers, the two taught with the blind
spot included. The word beside the blind spot is given back right by
the first way in every case. So today nothing is filled: with half of
its forms present a known word gives nothing back at its place.
`test_item18_expect.py::test_81_expectation_restores_the_omitted_native_word`
shows that the expected field does hold the omitted word's concept; it
does not reach the reconstruction. Two reasons were read in the source:
the way back from the open field is scored on the visible bytes only
(`Omission.visible_events`), so nothing teaches it to fill; and the
expected field is written beside the field, not into what the way back
reads. Why a word with half of its forms present gives nothing at all
was not traced: its basic percept is read by the share of its two
boundaries that is present (`BasicPercepts.read_points`), and both
boundaries are required (`read_collection`).

**The rule** (the form is Claude's; it carries §19.22's way c and way b
from a proposal to a rule, since Alec asks for what they give).

30. *The spread reaches the first layer, and the way back reads what it
    expects.* Rule 23's fan and share run on the Pi of order zero, the
    mereology's own entries, as they run above it. Up: a part that is
    present gives its value to the rows that read it, by its fan. Down:
    a row gives its value to its parts, by share, so a part that is
    missing receives a share from the rows its neighbours raised. This
    is the Pi generalizing over neighbouring elements. It writes
    expected percepts, as rule 25 writes the expected field, and the
    evidence is unchanged (rule 27); the row itself still requires its
    members (rule 2). At the reconstruction, where a look's percepts
    were withheld the way back reads the expected ones in their place
    (*default*). This amends rule 23, where the way down stopped at
    order zero's concepts. *Default:* a concept reaches its row directly
    (rule 21: one concept for each row), so the Sigma of order zero and
    its combined vector (rule 17) are not passed on the way down.

*Checked as arithmetic, not as a toy of this case.* §19.11's toy has
the same step one order up: a row read at .8 by two members, one
present, raises the row, and the row's share gives the absent member
.33, where the exact way back gives it nothing. Whether the share alone
is enough to give a word's bytes back, with five or six of its eleven
forms present, is what the testpoint measures.

**The testpoint** (certificate 24; a class `TestBlindSpot` in
`test/test_testpoint.py`, which prints what it measures as that file's
other classes do; built with step 4 of §19.21's order).

1. *The lesson.* The small lesson model (`test_item18_expect.lesson`),
   two sentences of two words, taught with everything visible until
   each word's row exists (rule 20: at least `concept_mint_count`
   readings).
2. *Inside a word.* Every second form of the second word is withheld
   through the omission's own draw, the sentence is read with no
   teaching, and the reconstruction is decoded. It passes if all the
   word's bytes are right at its place and the first word's bytes are
   unchanged.
3. *Not invented.* The same with a sentence whose second word was never
   taught: its place stays empty, and no taught word appears there.
4. *A whole word, reported and not required.* The second word wholly
   withheld, after a lesson taught with that omission: the bytes right
   at its place are printed. Filling it needs the word beside it to
   expect it, which rule 30 gives only where a row holds the two
   together (§19.22, the cost of one mechanism).
5. *Printed for each case:* the text, the bytes withheld, the decoded
   reconstruction, the bytes right at the blind spot and beside it, and
   the expected value of the withheld forms against the largest
   expected value of any form that was never present.

*On today's source it fails* at point 2 (0 of 5). It is a mechanism
certificate of item 1.7 and must pass there. *Decided* (Alec,
2026-11-17, asked whether it should go into the suite now, marked as
expected to fail: "With item 1.7"): it is added with item 1.7, and not
before.

### 19.24 A signed value is replaced by its two unsigned values, on the way up and on the way back (Alec, 2026-11-18)

Alec, 2026-11-18, of item 1.8's failures over signs (1.8 §68.62 and
§68.63):

> I do not know the fix for the current code, but let me speak to what
> should happen in the final version.
>
> You said "Since then a read of -1 pushes minus the whole code: the
> form and both lanes below zero". The sigma may be negative, but in its
> use in higher orders it travels through symbols, and in lower orders
> it travels back through percepts. In both cases it should be replaced
> by the two unsigned symbols or percepts that represent the vector: so
> 0.5 becomes [0, .5] and -0.5 becomes [.5,0]

**The rule.** A Sigma's value is signed, and the sign stays inside the
Sigma. Going up, what the symbols hold of it is two unsigned values.
Going back down, what the percepts hold of it is two unsigned values.
In both, its positive part is the one and its negative part the other:
.5 is against 0 and for .5, and -.5 is against .5 and for 0. No symbol
and no percept holds a number below zero.

**Where it stands in §19.21.**

- *Up.* Rules 3, 4 and 10 state it already (§18.25, "Yes, one number").
  Certificate 3 gains Alec's numbers.
- *Back down.* New in the record. Rule 17 said only that the Sigma
  returns the combined vector. It gains the sentence above.
  *Claude's reading of where that is built:* at order zero the combined
  vector that the Sigma returns is split, coordinate by coordinate, into
  the for vector and the against vector of rule 7; above order zero each
  symbol's signed input is split into that symbol's pair. The share down
  of rules 23 and 30 passes a Not entry to the other lane; what it
  moves is the two lanes, unsigned.

*Claude's reading of the order in which the pair is written:* the
negative part first. The stored code writes the for lane first
(`MeaningCodes`), so in the code's order .5 is (for .5, against 0). The
layout does not change.

**What today's source has in its place** (1.8 §68.63). A symbol carries
a row and one signed number. A word's two lanes are filled by the
attention walk's own pair, read through the fold, and the readout
`IndexedSigmaConceptsFromPercepts` gives one signed number of which
only the size is kept. Rules 8, 10 and 12 replace these, and the two
rows added to the table of readers in §19.21 name where. Item 1.8's
repair stores what the committed source stored, and does not take the
pair from the readout's sign: tried there, a word stored wholly against
met a reader that is not ready for it.

*Open for this item, and not yet asked of Alec:* what becomes of the
attention walk's own pair once the one Sigma's value is the pair. Today
the two can differ: in certificate 81 the walk says against of a first
word that is present, in most presentations, while its readout is above
zero (1.8 §68.63; observed, not traced). *Default:* the walk's pair
goes with the fold that reads it (rules 8 and 12), and the pair of a
word is the Sigma's alone. It is point 5 of the open list in §19.21.

### 19.25 Words, objects, then their meta concepts: the order of learning, an example, and two testpoints (Alec, 2026-11-18)

Alec, 2026-11-18 (his messages in full: 1.8 §68.67), of how a word and
its object come to be tied:

> perhaps many of the ties between word and object will have to be done
> first in a multi-modal concept, where "Alec" and "he" have a common
> perceptual basis or concept in another modality, so that we can tie
> those words to that concept/stimulus. as cited by Walker Percy, it is
> not until Helen Keller experiences water at the same time that she
> feels water running over her hand that she learns this identity
> between water-word and water-object.

> so we learn word generalizations, and we learn object generalizations,
> and THEN we learn word-object meta-concepts, because those are
> concsistently associated. there isnt a statistic that will magically
> produce identity across independent tokens.

> Please make sure this example of meta-concept learning is in the doc.

> And make sure we have a testpoint for cross-modality hebbian learning
> as just described.

> In addition to pavlovian bell-salivate sequence learning.

And of place and time, in the same answer: "they are necessary at the
level of events: there may be a one-shot learning that remembers a
percept and its time/place. So I think it is really generalization that
should remove that information".

**The rule.**

31. *Words, objects, then their meta concepts* (Alec). In this item's
    terms; each clause is *Claude's reading* and names the rule it
    rests on.
    - *A word generalization and an object generalization are each
      learned from their own occurrences.* Each is a row that MTM
      nominates on recurrence, with its concept (rules 20 and 21). The
      object's may be of another modality than the word's.
    - *An occurrence keeps its place and time; a generalization does
      not hold them.* An occurrence is a member that MTM holds, or an
      entry of LTM0, and a single one can be remembered with its place
      and time. What is made from occurrences stores neither: a symbol
      stores none (rule 13), and a concept's stored code has none. No
      reader removes them by hand; today four do (1.8 §70).
    - *Then the meta concept of the two, by their consistent
      association.* It is made when the word's concept and the object's
      concept have been present together on enough occasions. It is one
      order above the higher of the two and reaches both in one step
      (rule 26). After it a word that is found raises its object, and an
      object that is met raises its word (rules 23 to 25).
    - *Nothing joins independent tokens.* Likeness of codes pairs
      nothing (rule 16), and no count over occurrences that share
      nothing makes an identity.
    - *Two words are joined through what both are joined to* (Alec's
      "Alec" and "he"): a perceptual basis or a concept, which may be of
      another modality.

*Open, for Alec: how it meets rule 20.* Rule 20 has `interpret` make a
word's meta concept at once, with a blank object where the word has
none (2026-11-17: "if we interpret a word without an object, we must
create a blank one"). Rule 31 has the meta concept learned after the
object exists.

- *Claude's reading, recommended:* both stand, and they meet at the
  blank object. The meta concept that `interpret` makes is the place.
  What consistent association learns is the object in it: an object
  generalization that has been present with the word on enough
  occasions takes the blank object's place in that meta concept.
- *The other reading:* the word then has two objects, the blank one and
  the learned one. As the choice among a word's objects stands
  (`InterpretLayer.preferred_objects`: what the context expects, then
  the higher order) it would take the blank one, which is one order
  above the word (*default* of rule 20), over a perceptual object of
  order zero.

*Answered in direction by Alec the same day: §19.27.*

**What the source does today** (read on 2026-11-18 in item 1.8's
working tree; not run, except where a test is named).

- *A word and a percept of another modality.* The scalar modality
  (`bin/ScalarPercepts.py`: "A continuous native modality beside text")
  is perceived beside the text. At a kept close `ScalarPercepts.commit`
  pairs every word of the sentence with the scalar percept present with
  it, and `PerceptCollections.observe` collects the pair in MTM under
  the word. At `concept_mint_count` occurrences (3 in `data/model.xml`)
  it mints a concept of order zero, a percept collection, writes the
  percept into it, and writes the word's definition
  (`InterpretLayer.define`). So a word is already tied to what it
  recurs with, across modalities, by recurrence in MTM. Tests:
  `test_item18_word_trust.py::test_88_scalar_definition_reads_one_after_the_same_six_views`,
  `test_item18_recognition.py::test_scalar_percept_and_its_collection_restore_with_their_typed_references`,
  and the two scalar curricula under `RUN_SLOW`.
- *Where it differs from rule 31.*
  (a) The object is not learned on its own first: the collection is
  made for the word, out of the word's occurrences (MTM's entry is
  found by `defining_word`).
  (b) Every word of the sentence is paired with the percept. What tells
  a consistent pairing from a chance one is the collection's trust,
  agreement over views, read afterwards.
  (c) A word has one perceptual definition: every percept it meets
  goes into the one collection, and two raise ("a word has more than
  one perceptual definition").
  (d) Two words that meet the same percept get a collection each, not
  one object.
  (e) A word alone does not raise its object: no spread from a word to
  its object exists (1.8 §59.2). Rules 23 to 25 build it.
- *A sequence.* `expect`, the map at order zero, is taught by omission
  alone (`test_item18_expect.py::test_82_omission_is_the_only_teacher_and_consumers_detach`)
  and restores the omitted second word of a taught pair
  (`test_81_expectation_restores_the_omitted_native_word`). That is one
  word expected from another within one sentence. Whether anything is
  expected across two sentences was not measured. Since item 1.8
  nothing estimates a sentence (1.8 §68.53). `doc/Orders.md` has
  order-zero states "joined by succession make a sequence, an extended
  event" in MTM (Alec, 2026-11-12); whether the source builds such
  sequences was not read.

**The example, for the documentation.** *Placed on 2026-11-18 in
`doc/Language.md`, under "What `interpret` creates", once the other
documentation pass was on `main`.*

> *How a word comes to mean: the well-house.* Helen Keller could feel
> the letters w-a-t-e-r spelled into her hand, and she could feel
> water. For weeks the two were unrelated: a pattern of touches that
> she could repeat, and a wetness that she knew. At the well-house her
> teacher spelled the word into one hand while water ran over the
> other, and the two became one thing (Keller 1903; Percy 1975).
>
> Three things were learned, in this order. First the word: its
> spelling recurred, and the recurrence made a word concept. Second the
> object: the feel of water recurred, in another modality, and made an
> object concept. Each is a generalization over its own occurrences.
> Each occurrence has its place and time, and the generalization holds
> neither. Third, and only then, the meta concept that joins them: the
> word concept and the object concept were present together, and that
> consistent association is what was learned.
>
> Nothing else would have joined them. Alec, 2026-11-18: "there isnt a
> statistic that will magically produce identity across independent
> tokens." Two words are joined in the same way, through what both are
> joined to. "Alec" and "he" are one person because both are tied to
> one perceptual basis, and not because their occurrences are alike.

*Claude's note, from memory of the essay, to be checked at the doc
pass.* Percy sets two things apart: a signal that is followed by
something, as the bell by the food, and a word that is the name of a
thing. The two testpoints below measure the two: the sequence at order
zero, which an animal has, and the meta concept one order above its
word and its object, which needs symbols (rule 29).

*References* (supplied by Claude from memory; to be checked at the doc
pass). Keller, H. (1903), *The Story of My Life*, Doubleday, Page.
Percy, W. (1975), "The Delta Factor", in *The Message in the Bottle*,
Farrar, Straus and Giroux. Pavlov, I. P. (1927), *Conditioned
Reflexes*, Oxford University Press. Hebb, D. O. (1949), *The
Organization of Behavior*, Wiley.

**The testpoints** (certificates 25 and 26; classes in
`test/test_testpoint.py`, which print what they measure, as that file's
other classes and `TestBlindSpot` do. *Default:* built with item 1.7
and not before, as Alec decided for `TestBlindSpot`, and not marked as
expected to fail. The class names are *defaults*.)

*Certificate 25, `TestCrossModalTie`: a word and an object of another
modality are tied by their consistent association.*

1. *The word alone.* A word is read `concept_mint_count` times with
   nothing else present. Its row and its concept exist. It has no
   object but the blank one.
2. *The object alone.* A stimulus of another modality (*default:* the
   scalar modality) is presented as often with no word. Its row and its
   concept exist, and no word is tied to it. The row holds no place and
   no time; each occurrence that MTM holds keeps its own.
3. *The two together.* After the same number of presentations of the
   word with the stimulus, the meta concept exists over the two
   concepts, and the word's object is that concept.
4. *Each raises the other.* The word alone leaves the object's concept
   expected above zero, and the way back gives the stimulus's percepts
   at their place (rule 30). The stimulus alone leaves the word's
   concept expected above zero.
5. *No tie without consistency.* A second word, presented as often,
   each time with a different stimulus, is tied to none of them.
6. *Two words, one object* (Alec's "Alec" and "he"). A third word is
   presented with the first stimulus. It is tied to the same object
   concept, no second object is made, and each of the two words then
   leaves the other's concept expected.
7. *Printed for each stage:* the rows and concepts that exist, the
   word's objects, the expected value of the object from the word and
   of the word from the stimulus, and the largest expected value of any
   concept never presented with either.

*On today's source* (by reading, not run): point 3 holds in the form
the source has it (a percept collection and a definition after three
pairings); points 2, 4, 5 and 6 do not, for the reasons (a), (b), (d)
and (e) above.

*Certificate 26, `TestBellSalivate`: a sequence is learned; the bell,
then the food.*

1. *Before.* The first stimulus alone leaves the second not expected.
2. *After pairing.* The first has been followed by the second, with
   what the second always brings (the response), in N presentations.
   The first alone then leaves the second expected, and the response
   through it.
3. *Order, reported and not required.* The second alone, and the value
   it leaves for the first.
4. *No tie without the sequence.* A third stimulus, presented as often
   and never followed by the second, leaves it not expected.
5. *Extinction, reported and not required.* The first presented alone
   M times more, and the value it then leaves for the second; the
   omission is what teaches (rule 29).
6. *Two forms of the sequence, both printed:* the two stimuli as
   successive words of one sentence, which is item 1.8's certificate 81
   on other words, and as two sentences in succession. The second form
   is required only once its mechanism is decided (open point 7 of
   §19.21).

*On today's source:* the first form passes (certificate 81). The second
was not measured.

### 19.26 A felt world for the math corpus: words matched to feelings at places (Alec, 2026-11-18; a todo)

Alec, 2026-11-18:

> Please add this todo also: for the math learning corpus, we should
> make a model test that matches words to objects, where objects are
> feeling-intensities within a 3-D space. So a sentence like "one was on
> top of two", even though it doesnt make a lot of sense, will still
> describe a world that is phenomenally accessible to the model, an in
> which the feeling of one is actually above the feeling of two. That
> also forma a basis for the identity tests that we are talking about
> now. Maybe girls are even and boys are odd, so we can map a gendered
> language onto numerals. The point is that we can give our math corpus
> perceptions that are associated with intensities, just as the large
> models already do when feeding in billions of images of our faces and
> homes.

**What it asks.** A test on a model, for the math corpus, in which the
words have objects that the model perceives. An object is a felt
intensity at a place in a space of three dimensions. A sentence of the
corpus then describes a scene the model can perceive: in "one was on
top of two" the feeling of one is above the feeling of two. The same
world is the basis of the identity tests: certificates 25 and 26 of
§19.25, and the match of 1.8 §70. His "maybe": a property that can be
perceived carries a gendered vocabulary, even numerals as girls and odd
ones as boys.

**What exists** (read on 2026-11-18; not run).

- The feeling modality, `bin/ScalarPercepts.py` ("A continuous native
  modality beside text"): one felt value present with a sentence,
  perceived by its level against part and whole percepts (Alec,
  2026-11-12), at one fixed place (`where_registry.slices['scalar']`).
  It has no place axis of its own.
- A modality has its own number of place axes (Alec, 2026-11-13): text
  one, an image two.
- The tie of a word to a felt value by recurrence
  (`PerceptCollections.observe`), with its differences from rule 31
  (§19.25).
- The generated math corpus and its tests (`test/test_math_chain.py`).
  Its numerals are opaque words, and nothing computes on a numeric code
  (Alec, 2026-10-07). Its sentences have no scene.

**What it needs** (*Claude's reading; it is to be specified, and
nothing is designed here*).

1. The feeling modality with three place axes: a felt intensity at a
   place.
2. For the corpus's sentences that describe an arrangement, the scene
   each one describes, generated with the sentence: which intensity is
   at which place.
3. The tests. A numeral's word is tied to its felt intensity
   (certificate 25 on this world). A word of place ("on top of") is
   tied to the relation of two places. Two words with one perceptual
   basis are tied to one object (his gendered words on even and odd
   numerals). A mention is matched to the object it describes by the
   one match of 1.8 §70.

*To keep* (*Claude's note*): the numeral stays an opaque word and the
feeling is its object. The grammar still computes nothing on a
magnitude (Alec, 2026-10-07; and 2026-11-13: "percept activations have
a kind there but no magnitude").

**Its place** (Alec, 2026-11-18): "Then 1.6 becomes the number
modality, the math corpus (it needs more development), and the work
you called out as "unnumbered: felt world for the math corpus" which
combines both." So this is part of item 1.6, which follows item 1.7,
whose rule 31 and certificates 25 and 26 it gives a world to. The
work that had the number 1.6 until then is the first part of item
1.5.

**Open.** Whether
"above" is one of the three axes by convention or is learned from the
scenes. What carries even and odd as something perceived. Whether a
scene is present with every sentence, as the felt value is today, or
only in a lesson.

### 19.27 Alec on §19.25: the bootstrap, two other ways to learn the tie, an existing object first, and the blank (2026-11-18)

Alec, 2026-11-18, of what the source does today (§19.25) and of open
point 6. Of "A word that recurs with a felt value gets an object made
for it after three pairings; the object is not learned on its own
first":

> [It] is one option. We can do that explicitly with a relation A->B in
> the taxonomy with zero trust, or perhaps learning the possible
> association when none is known by using pi-not-sigma every time, but
> only promoting when PI indicates. That strategy looks like the
> existing conceptual hierarchy, so it may make sense to leverage that,
> but we would beed a better method for managing certainty and we would
> need a forgetting over that corpus (the later being the reason I
> suggest a small MTM with forgetting, so that promoted concepts are not
> forgotten, just more or less certain depending on their truth or
> falsity when active).

> That is a bootstrap because we cannot know the object expect through
> word for a unimodal model. It is not a great solution, but it may work
> with sufficient complexity (it works for LLMS, after all).

Of "Two words that meet the same percept get an object each":

> We should probably check to see if an object-concept already exists
> before creating a new one. Please explain the concept here though,
> maybe this is necessary (but I think the generality of pronoun lookup
> will eventually correct this).

Of open point 6, the blank object:

> better never to have the blank at all, but to avoid a discontinuous
> decision, we might train both and remove the blank only if the vectors
> subsequently overlap in conceptual space.

(His arrow in the first passage is written "->" here, and "[It]"
stands for the sentence he quoted.)

**What it says** (*Claude's reading* where marked).

1. *The bootstrap stays.* A model with one modality cannot know an
   object except through its word. So an object made for the word, out
   of what the word recurs with, is how it starts.
2. *Two other ways to learn the tie are named, and none is chosen.*
   (a) A relation from the word to the object in the taxonomy, written
   at once with zero trust. (b) The association learned by Pi, Not,
   Sigma each time none is known, and promoted only when the Pi
   indicates. The second is like the concept hierarchy that exists and
   may be built on it. It needs a better way of managing certainty, and
   forgetting over the candidates: a small MTM with forgetting, in
   which what is promoted is not forgotten and is only more or less
   certain, by its truth or falsity when it is active.
3. *Look for an existing object first* ("We should probably"): before
   an object is made for a word, whether an object concept exists.
4. *The blank object.* Better never to have it. To avoid a
   discontinuous decision both may be trained, the blank object and the
   learned one, and the blank removed only if their vectors come to
   overlap in conceptual space ("we might").

**The explanation he asked for: why two words get an object each
today, and what looking first would change.**

Today the object that a word gets is that word's own account of what it
has been met with. `PerceptCollections.observe` keeps one MTM entry for
each word (found by `defining_word`), adds to it every percept that is
present with the word, and on the third occasion makes a collection of
it and defines the word as that collection. Nothing looks for a
collection that already holds those percepts. So if "Alec" is read
three times with one percept, and "he" three times with the same
percept, there are two collections with the same one member, each named
by one word.

That is what lets words be told apart afterwards. Every word of a
sentence is paired with the percept that is present. In "this is
water", read with the feel of water, "this", "is" and "water" each
start a collection with that member. They come apart by what each goes
on to meet: "is" meets everything, its collection widens, and its trust
(agreement over views) falls; "water" keeps meeting the same thing. Had
the three been given one object on the first occasion they would be one
concept, and parting them would be the discontinuous decision.

It is also why two words for one thing are not joined. "Alec" and "he"
keep an object each, with the same content. A match of the one object
against the other finds them the same: both numbers of 1.8 §70.9 are 1.
That is his "the generality of pronoun lookup will eventually correct
this".

*What looking first would change* (*Claude's reading*). The second word
is tied to the first word's object. That is right for "Alec" and "he".
It is wrong for "is" and "water", unless the tie itself can be doubted.
So looking first needs his way (a): the tie is a relation that starts
at zero trust and earns it by consistent association, and the object's
own certainty is not touched by a doubtful tie. Today the certainty is
kept on the collection (`ConceptTrust`: views and agreement), and the
word's definition (`InterpretLayer.define`, a part row of LTM1 with its
two truth lanes) is written when the collection is made.

**What exists for way (b)** (read on 2026-11-18; not run). The one
writer of a concept's further inputs, `ConceptHierarchy.observe_member`
(rule 21). Certainty as the length of the for lane (`ConceptTrust`;
Alec, 2026-11-06). MTM's candidates, which lapse after `mtmHorizon`
(16) closes without a reference, in a pool of `mtmCapacity` (64), and
are promoted at `conceptMintCount` (3). Rule 20, under which MTM
nominates every row. And Alec, 2026-11-17: "The form of MTM can
probably be the same as percepts or symbols ... an easily promoted and
easily forgotten store".

**Open, for Alec; none of it blocks item 1.8.**

- *"Remove the blank"* meets two earlier decisions: "Let's keep
  concepts forever; only memories get retired" (2026-11-16), and
  "forgetting should probably not forget definitions" (2026-09-29).
  *Claude's reading, to confirm:* the word stops reading the blank
  object; the blank's row is left and is not reused.
- *What "overlap" is.* *Default:* the match of 1.8 §70, with one of its
  two numbers at one.
- *Which way the tie is learned by in this item.* *Default:* the
  bootstrap as it is, since he says it stays, with way (a)'s trust on
  the tie wherever an existing object is found first.

### 19.28 A look is a place that returns percepts; a Pi finds the spellings; hit and miss set its width (Alec, 2026-11-18)

Asked when a word found by a near miss is built, and what "close"
compares (1.8 §68.76), Alec, 2026-11-18:

> A look should be a .where that returns percept indices. If that is
> forced to words, then the lexer is doing the models job. Caat and cat
> both match on whole of those letters, and they share many of the same
> parts, so a pi in that neighborhood should find different spellings,
> and return the basic percept corresponding to all of them (although
> that pi lookup happens in 1.7).
>
> And the hit/miss on a word will hopefully feed into the width of that
> pi hypercone.
>
> So that logic probably goes in 1.7

**What it decides.** It is built in this item. It is rule 32 of §19.21.

1. *A look is a `.where` that returns percept indices.* It is not made
   of the stem's word intervals. Where the cut into words is made before
   the model reads, "the lexer is doing the models job".
2. *A Pi finds the spellings.* "caat" and "cat" match on the whole of
   those letters and share many of their parts. A row of the Pi in that
   neighbourhood finds the different spellings and returns the basic
   percept that corresponds to all of them. This is the lookup over an
   inexact match: no second lookup is added for it.
3. *Hit and miss set the width.* Whether a word was found or missed
   feeds the width of that row's cone.

**What the record has to build it on.**

- A row of the Pi is a hypercone, and its value for an input is the
  input's centrality in the cone (§19.21, the Pi layer; Alec, §18.14:
  "a (sparse) hypercone on hypercube parts and wholes").
- A row exists when MTM nominates it (§19.10), and rows are taken as
  neighbours by the mereology (rule 23 and §19.11).
- Rule 30: the spread reaches the first layer, and the way back reads
  what it expects.
- Until this item, in the source that item 1.8 leaves: a look is made
  of whole word intervals from the stem, the lookup of a form is exact
  (`InterpretLayer.word_for_form`), and a word read with both lanes is
  not offered to `interpret` (1.8 §68.76).

**Not designed here, to be specified with the item.**

- What a look's `.where` is when it is not word intervals: any
  interval of the input, or the extents that attention's acts can
  reach; and what then becomes of the stem's word intervals and word
  texts, which `stage_input` and the first look read today.
- Which quantity of a row is "the width" of its cone, and what moves
  it: *Claude's reading* is that a miss narrows and a hit widens or
  keeps, by the cost of the grammar's choice of `interpret`. He says
  "hopefully"; it is not a rule yet.
- What a hit and a miss are: the lookup finding a held word or none, or
  the sentence then being read rightly or wrongly.
- Whether a letter's place counts in the match, since in "caat" the
  last letter stands one place later (rule 31: a generalization holds
  no place).

### 19.29 "There is a word there": inclusion in a meta concept (Alec, 2026-11-18)

Alec, 2026-11-18, deciding against a call of `interpret` at every word
(1.8 §68.79): "interpret() should only interpret the word as an object
if 1) there is a word there (as indicated by inclusion in a meta
concept) 2) there are not multiple words there."

*What exists.* The second condition is built (`Attention.narrowing_mask`:
the bracket covers one word). The first is not what the source tests:
a word "is there" when its form was admitted and has a word concept
(`known`, `InterpretLayer.word_for_form`), whether or not a definition
row (word DEF object, `InterpretLayer.define`) names an object for it.
On a fresh model no admitted form has such a row, and a word gets one
only through being read (`InterpretLayer.reading` holds its name in
MTM; the row is written at a kept close).

*Placed here by Claude, not by Alec:* item 1.8 keeps today's test, since
requiring the row now would read no word on a fresh model. With rule 31
a meta concept is learned from the consistent association of a word
generalization and an object generalization, so a word can be in one
before it is ever interpreted, and the condition can then be tested as
he wrote it.

*Unspecified:* whether the test is the definition row or the row MTM
nominates for the tie; what a word with no meta concept is in a tree,
if anything (the false branch of `InterpretLayer.reading` returns the
word concept itself); and how `interpretBias` (1.8 §68.79, set to 1) is
lowered "when doing so is valuable based on previous experience".

### 19.30 `interpret` takes the largest word that fits the look (Alec, 2026-11-18; an idea)

Alec, 2026-11-18: "an an idea for interpret: take the largest word that
fits in the .where (even if it is composed of smaller words)."

*In plain terms.* If the look covers "ice cream" and the model holds a
word for "ice cream" as well as words for "ice" and "cream", `interpret`
takes "ice cream". If it holds no such word, the look holds two words,
and attention has to narrow before either is interpreted.

*What it amends.* His second condition of the same day (§19.29, 1.8
§68.79), "there are not multiple words there", then counts words at
their largest size: a look that is one large word is one word, whatever
smaller words it is made of.

*What exists.* The idea is in the record from the other side, as the
dynamic stop (`doc/FutureWork.md`, "The dynamic stop: glossing above
words, descending at novelty"): a known unit of several words that
reads purely at its wider bracket is read as one symbol by `interpret`.
It is not built. `Attention.narrowing_mask` offers `interpret` only
where the bracket covers one of the stem's word intervals, the lookup
(`InterpretLayer.word_for_form`) takes one such interval's form, and
1.8 §60.1 says "A span of several words taken as one unit is not
built". A phrase that recurs can already become a concept defined by
its parts (`conceptMintCount`).

*Placed here by Claude, not by Alec:* with rule 32, since it needs a
look that is not forced to the stem's word intervals, and a Pi that
returns the percept for what the look holds. It is in no text of item
1.8.

*Unspecified:* what "largest" measures (the extent, or the number of
parts); what is taken when two large words overlap and neither holds
the other ("New York" and "York City" in "New York City"); whether the
smaller words stay reachable by a further `focus` after the large one
is declined or interpreted; and what makes a unit of several words a
word (a meta concept of its own, §19.29).

### 19.31 A look may carry a content condition: priming a whole, with the place (Alec, 2026-11-18; a direction)

Alec, 2026-11-18, after the review of the loop of looks (1.8 §68.81):

> My guess is that letting the architecture attend and supply .where
> coordinates will work better eventually, but will require more
> training up front. Allowing it to do a bit of top-down priming for a
> content match would probably work better than just supplying .where;
> like, if it can say " I want percepts that have a whole that consists
> of a set of letters and no spaces, numbers, or punctuation" in
> conjunction with .where basically limits reading to words, but still
> gives the architecture the ability not to filter by content. So
> there is a top-down conditioning based on expectation or desire to
> read a word. Can we allow that easily?

**In plain terms.** A look today is a place and nothing else. He adds
a second thing a look may say: which kind of whole it wants there. On
"red cat sleeps", a look at the whole line that wants "a whole of
letters only" gets three things back, `red`, `cat` and `sleeps`; the
same look with no wish gets the line as it is, spaces included. The
cut into words is then something the model asks for, and can leave
off, where today the stem makes it before the model reads.

**What exists (read in the source at `bfd2c606a3`).**

- *The content he names is a row.* WholeSpace holds nine canonical
  whole properties (`_CANONICAL_PROPERTY_ROWS` in `bin/Spaces.py`):
  letter, digit, whitespace, punctuation, capital, control, high byte,
  pad and word. Row 8, word, is fixed: its members are exactly the
  letters and cannot be taught
  (`PrimitiveProperties.define_word`, "An immutable whole over exactly
  the reader's letter predicate"). Row 0, letter, is of the same kind and
  is taught; it is the property whose membership drifts in 1.8 §68.81.
- *The wish is a concept, and its priming is built.* A frozen concept
  named "reading", and `BaseModel.set_reading(desire, valence)`: it
  desires that concept and, while it is on, desires each batch's
  word-whole rows in WholeSpace ("the hard-coded concept ->
  word-isolating-wholes wiring"). Desire and expectation are one
  surface (`relevance_weights`: one plus the expected field plus
  desire). Nothing calls `set_reading` today but its test
  (`test/test_frozen_concepts.py`).
- *The place is passed down.* A look writes `_passback_scope_where`,
  and `Models._passback_scope_ps` is described as retargeting
  perception by "native percept identities, which choose the
  mereological units, and field brackets, which choose the region".
  *Not checked:* whether that second route runs anywhere in the loop
  of looks.
- *The read takes many brackets at once.* `cs_read_memberships` reads
  one field for each bracket it is given; the first look gives it
  every word bracket (`FamiliarTier.open_sentence`).

**The gap.**

1. *The cut into words is made by the stem and is not asked for.*
   `Meronomy.word_spans` takes the maximal runs of letters,
   `WholeSpace.stage_analysis_spans` cuts the input by runs of one
   type, and `pin_word_spans` pins the words. Attention's brackets are
   made of those intervals. This is "the lexer is doing the models job"
   (§19.28).
2. *A look's read takes a place and no condition.*
   `SentenceLooks.perceive` gives `cs_read_memberships` the look's
   extent as one bracket. Desire and expectation reach `attend` as a
   signal (1.8 §60.1, rule 9) and do not shape what a look returns.
3. *So a look at several words reads nothing* (1.8 §68.81, finding 2):
   no whole holds on every byte of an extent with a space in it.

**What it would be** (Claude's reading of his words, to be specified
with rule 32).

- A look is a place and, optionally, a primed whole. With a whole
  primed, the look returns the maximal runs inside the place on which
  that whole holds, each read as its own bracket. With none, it
  returns the place as one extent, as now.
- With row 8 primed, those runs are the stem's word intervals by
  construction, since row 8 and `word_spans` use one predicate
  (`Meronomy._letter`). So the first look and every later look become
  one operation, a look at several words reads its words in parallel,
  and the stem's pin can become a check and then go.
- The wish is on from the start (`set_reading`, as `interpretBias` is 1
  from the start), so reading words need not be learned from nothing,
  and it can be lowered or chosen later.

**How easily.** The read itself is small: `perceive` gives the read
the runs in place of the one extent, and the forecast of a look
already takes several readings. Larger, and rule 32's: attention's
brackets no longer made of the stem's intervals, and the stem's cut
removed.

*Placed here by Claude, not by Alec:* with rule 32. It is in no text of
item 1.8.

*Unspecified:* who sets the condition at a look (the standing desire
of "reading", the expected field, or a choice of `attend`); what a
graded priming does, since the surface is a number and a run either is
returned or is not; whether several wholes may be primed at once
(letters and digits); whether the condition names row 8 or the taught
row 0; and what a look returns for the bytes between the runs.

### 19.32 The condition as wires: a mask on what is active, asked for from above (Alec, 2026-11-18; a direction, in discussion)

Alec, 2026-11-18, on §19.31:

> Right now we are returning codes, but the efficient architecture is
> wires. We almost have this in the subspace class; the percept codebook
> in .what and the .active tell us what is in the .where that we have
> selected. Which means that we could learn a mask on .active by
> gradient. We might also be able to learn .where in the same way, but
> let's defer that unless you see a way to do that easily. So conceptual
> space asks for the kind of thing it wants (top down activation wires)
> and the perceptual space returns tokens of those types that occur
> within the .where span

**In plain terms.** A sentence's percepts are found once: which
percept each is (an index into the codebook), where it is, and how
active it is. A look then copies nothing. It is a mask over those
percepts: each is kept in proportion to being inside the look's place
and to being of a kind that conceptual space asked for. On "red cat
sleeps" with words asked for, the three word percepts keep their
activity and the two spaces lose theirs.

**What exists (read at `bfd2c606a3`).**

- *The form.* `SubSpace` holds `.what` (a codebook), `_index` (for each
  position, the row it selects: "the activation/selection IS the
  storage"), `.where`, `.when` and `.activation` (a strength for each
  position: "materialize() = event * activation").
- *The sentence's percepts, once.* `native_indices` and
  `native_part_spans` are two arrays with one index, a percept and its
  place (1.8 §74; item 1.5's percept pass). A `PerceptField` has an
  axis of events with their spans, and for a whole property such as the
  word row its events are the runs on which it holds.
- *A look today returns codes.* `BasicPercepts.read` builds a new field
  for each look: a containment test of each event in the bracket, a
  minimum over each concept's features, loops over rows on the host.

**What the mask would be** (Claude's reading).

- For event `e`: its activity, times a place factor (its span lies in
  the look's `.where`), times a type factor (the request's value at the
  event's row, a gather). With no request the type factor is one.
- His example has two lanes: letters wanted, and spaces, numbers and
  punctuation not wanted. A request is then a pair of lanes over the
  rows, as evidence is.
- The request is activity and not position, so it does not cross the
  line of the spaces (2026-11-15: no space moves or scores another's
  positions; activations may pass).
- The designed way down is rule 30: the first layer's own matrices read
  from the concepts to the percepts they expect. The request is that
  reading, with the frozen concept "reading" (§19.31) as its source for
  words.

**Points that need his word.**

1. *Which wires take the gradient.* Today `expect` has one teacher,
   omission, and what reads its field is detached
   (`test_82_omission_is_the_only_teacher_and_consumers_detach`). A mask
   taught by the cost of what the look reads is a second teacher on the
   way down, unless it lands elsewhere (desire, or `attend`'s own map).
2. *What a returned token is.* Each returned percept as its own reading
   (the words of a look in parallel, as the first look has them), or
   one reading pooled over the place (as a later look has today, which
   is why a look at several words reads nothing, 1.8 §68.81).
3. *Graded.* The mask is a number for each event, so a half-asked kind
   returns half its activity; whether that is wanted, or what turns it
   into returned and not returned (the open point of §19.31).

**The place by gradient (deferred by him).** One form is easy to
state: the place factor is the same operation on the event's `.where`
code that the type factor is on its `.what` row, a match with a place
asked for from above, which has a gradient in what is asked. *Not
checked:* whether the `.where` code is smooth enough for that to teach
anything (a containment test has no gradient in its bounds). Today the
place is a discrete choice of `attend`, taught by its lesson and its
departures.

*Placed here by Claude:* rule 32 and rule 30 of this item give the
request and the look; the wiring is part 1 of item 1.5. It is in no
text of item 1.8.

### 19.33 Asking for concepts: the translation to percepts as the filter (Alec, 2026-11-18; in discussion)

Alec, 2026-11-18: "Can we also filter by asking for certain concepts
and relying on the translation to percepts to act as a filter?"

**Yes, and an earlier form of it is in the source** (read at
`bfd2c606a3`).

- *The translation.* `ConceptualSpace.cs_percept_attribution`:
  "Attribute each requested pole through definitions to percept
  events." A demand over concepts, on both lanes, is read back through
  the definitions of order zero (`cs_reverse_presence`), shared over a
  concept's written basic percepts by their weights, and placed on the
  events of the perception it is given ("Concept rows have no event
  coordinates; only the supplied perception can locate a demand").
- *The filter.* `Models._passback_scope_ps` takes the conceptual field,
  attributes it to the percept events, keeps what lies inside
  `_passback_scope_where`, and scales the activity of each part by
  one half plus one half of its strength: "Attention changes salience,
  never visibility. A half-strength floor preserves novel percepts".
- *Where it runs.* At a stage after the first in the loop over
  conceptual stages ("the top-down attention handoff"). The loop of
  looks does not call it. *Not checked:* whether it runs in today's
  lesson at all.

**What it is on an example** (read from the code, not run). Asking for the concept of `cat` on "red
cat sleeps": the demand goes down to the basic percept of the form
`cat`, that percept's event is at bytes 4 to 7, and the mask keeps that
token and lowers the others. It is a search for a thing already known.

**The gap, against §19.32.**

1. *It returns codes.* A new subspace is built with scaled content, and
   the attribution loops over rows on the host. As wires it is the
   first layer's matrix read backward (a request over basic-percept
   rows) and then the gather of §19.32.
2. *It never removes.* Its floor of one half is a rule of that design.
   His words of §19.32 ("returns tokens of those types") remove what was
   not asked for. Which holds is his.
3. *It goes down one layer only.* The demand is read through the
   definitions of order zero. Asking for a class or a meta concept
   needs the demand shared down the orders first, which is the spread
   of rule 30.
4. *It finds only what is written.* An exact way down returns the
   percepts of the definitions and nothing else (§19.22; §19.23: today
   0 of 5 bytes are filled at a withheld word). A word not yet known is not of
   any asked concept, and is returned only under the floor of point 2,
   or by a request for a whole (§19.31), which admits it.

**So the two requests differ in kind** (Claude's reading): a whole asks
for a kind of form and admits what is new; a concept asks for a known
content and is a search. Both are one mask.

*On the open point 1 of §19.32:* this translation runs through the
definitions' own weights, so a gradient through it teaches the
definitions unless it is cut there.

### 19.34 The Pi matches one percept tensor through what is active; no `materialize()` (Alec, 2026-11-18; in discussion)

Alec, 2026-11-18:

> Ok so we can do this, but it requires that there is a percept tensor,
> and the basic-percept-forming-pi that matches against it uses .active
> to selectively match that tensor. So we never want to call
> materialize() on the subspace.
>
> Agree?
>
> Then we can let the selection tensor coming top down from pi match
> wholes within the .where span to return a selected set.

**Agreed (Claude), and the look's read is already most of the way.**
Read at `bfd2c606a3`:

- *No `materialize()` on this path.* `SentenceLooks.perceive`,
  `FamiliarTier.open_sentence`, `BasicPercepts.read`,
  `PerceptField.read_percepts` and `LookForecast` do not call it. They
  take the sentence's arrays: the parts' percept indices and spans, the
  raw bytes, the primitive properties, and the spans of the wholes.
  (Elsewhere `bin` calls it at some 180 places, and the old pass-back
  of §19.33 is one of them.)
- *The percept tensor has an axis of events.* `read_percepts` joins the
  part events and the whole events on one axis, each with its span.
- *A mask over that axis exists, as a test.* `contained` is true for
  the events inside each bracket. It is made from the bracket, it is
  yes or no, and after it a new tensor of features by readings by
  events is filled for every read. That filling is the "returning
  codes".

**So the change is:** the sentence's percept tensor is made once and
stays; `contained` becomes the activity given to the Pi, a number for
each event that is the place factor times the request (§19.32); and a
look changes that activity and nothing else.

**Three conditions** (Claude's, from his earlier rules).

1. *What is not active is absent, not denied.* A masked event gives
   neither lane. Otherwise looking away from a letter would be evidence
   against it. This is his rule for `non` (2026-11-12: it withholds both
   kinds of evidence).
2. *The selected set is tokens, not bytes.* A mask over bytes that
   drops the spaces of "red cat sleeps" leaves one extent of letters,
   on every byte of which the letter whole holds. His wording has it
   right: the request matches *wholes* inside the `.where`, and what
   comes back is the set of events of that whole, `red`, `cat` and
   `sleeps`, each with its own span and each read by the Pi inside
   that span.
3. *Where the whole events come from.* Today they are the stem's runs
   of one type, cut by the taught memberships of the whole properties
   and then pinned to the fixed letter test (`stage_analysis_spans`,
   `pin_word_spans`). Under this form they are perception's candidates,
   found once for the sentence, and the choice among them comes from
   above. The pin is the part that is still fixed.

*Still open from §19.32 and §19.33:* which wires take the mask's
gradient, and whether a look may remove what was not asked for or only
lower it. His "return a selected set" reads as removing; he has not
said so.

### 19.35 How the request meets attention (Claude's reading, for discussion; Alec, 2026-11-18: "I don't know how this would integrate with attention")

Nothing here is decided. It is what the source suggests. *It answers
for attention as brackets; he meant attention as the filter that dims
what is outside its area, which is §19.36.*

**Attention already holds the mask** (read at `bfd2c606a3`). Inside
`Attention.narrow_step` a bracket is `covered`: for each word of the
sentence, whether the bracket holds it. The pooled keys and the three
field acts (and, or, not) are computed through it
(`reduce_field(..., mask=covered)`). Only at the look is the bracket
turned into an interval (`SentenceLooks.perceive`), which
`read_percepts` turns back into a mask (`contained`).

**The reading.**

1. *Attention gives the place, as now.* `attend` keeps its six acts
   (divide, descend or look at a word, unfocus, and, or, not) and
   chooses nothing new. The row of `covered` for the chosen bracket is
   the place factor of §19.32, handed to the Pi as it is.
2. *The request is not an act of attention.* It is the state of
   conceptual space, the surface of expectation and desire that
   `attend` already reads as a signal (1.8 §60.1, rule 9). The activity
   the Pi matches through is attention's place times that request.
3. *Attention's tests are taken on what comes back.* Today they count
   the stem's words in the bracket (`count`) and read their lanes. They
   would count the selected set.
   - One token, one lane: the grammar may call `interpret`. This is his
     "there are not multiple words there" (§19.29), and with a larger
     whole asked for it is "the largest word that fits" (§19.30).
   - Several tokens: attention narrows the place, as it does today for
     a bracket of several words.
   - No token: the place holds nothing of the kind asked. This is the
     cue he asked for as future work (2026-11-18: "feedback about the
     both/neither over symbols so we know if we should attend() by
     zooming in or out"), and it is graded: the summed activity of the
     set.

**On "red cat sleeps".** With words asked for, the first look returns
`red`, `cat` and `sleeps`; attention divides; the look at "cat sleeps"
returns two tokens, where today it reads nothing; the look at `cat`
returns one, and the grammar may interpret it. The acts are those of
today. With the concept of `cat` asked for, the first look returns the
one token `cat`, and it may be interpreted with no act of attention at
all: asking for a content is a search, and it saves the narrowing.

**What it leaves as it is.** The choice of place stays a choice of
`attend`, taught by its lesson and its departures. The request is
graded and taught, if at all, by gradient on the asking side. The two
do not share a teacher.

**Not known.**

- Who changes the request inside a sentence, if anyone. If `attend`
  must choose it, that is a new choice with its cost in training, which
  his proposal was meant to avoid. *Default (mine):* not `attend`.
- Whether attention's own reading of a bracket (the walk's keys and
  `native_word_poles`) and the look's reading can become one. They read
  the same place twice today. Not examined.
- Attention's mask is over the stem's words; the percept tensor's axis
  also has parts and other wholes. Which axis attention's brackets live
  on is rule 32's.
- If a look only lowers what was not asked for (§19.33), "one token"
  is no longer a count, and the test becomes graded, as the test for
  one lane already has to (1.8 §68.81).

### 19.36 With attention as the filter that dims what is outside its area (Claude's reading, for discussion)

Alec, 2026-11-18, correcting §19.35: "Sorry, with attention as the
filter that we talked about before that dims out-of-area content."
§19.35 answered for attention as brackets. This section answers for
the filter.

**The filter in the record.**

- `doc/Architecture.md`: "Attention is a mask over `.where`" (Alec,
  2026-10-07).
- `doc/Spaces.md`: "conceptual pass-back scales attention with a
  strictly positive floor; it cannot hide novel content."
- `Models._passback_scope_ps` (§19.33): a part's activity is scaled by
  one half plus one half of its strength, and the strength is zero
  outside `_passback_scope_where`. "Attention changes salience, never
  visibility."
- `architecture.attentionFloor`, 0.1, is still read and checked and is
  used by nothing since the loop of looks.
- The priming surface (`doc/Architecture.md`): "desire boosts, hate
  suppresses with floor 0 - suppression, never a veto".

**The integration: one activity, two factors, each with a floor.** For
each event of the sentence's percept tensor, the activity the Pi
matches through is its own activity, times a place factor (one inside
the look's area, the floor outside), times a request factor (one for a
kind asked for, the floor for the rest). Nothing reaches zero.

*On "red cat sleeps"*, looking at `cat`, words asked for, a floor of
0.1: `cat` is matched at 1, `red` and `sleeps` at 0.1, and the two
spaces at 0.01.

**What follows from it.**

1. *The factor is applied once to a token's reading, on both lanes.*
   Inside the conjunction of a word's parts a floor of 0.1 on each of
   five letters would give 0.00001, and on the against lane it would
   read as denial. Dimmed means less present on both lanes, never
   denied (§19.34, condition 1).
2. *The exact tests cannot read a dimmed field.* They count any value
   above zero (`narrowing_view`), which is how "Felix" at 0.0002
   against came to count as two lanes (1.8 §68.81). In a dimmed field
   every look holds every word at the floor, so every look would be
   "several words". Two ways:
   - *Two readings of the one activity (recommended for as long as the
     tests are exact).* The Pi, the surprise and the priming read the
     dimmed field, so a new thing outside the area is still seen and
     can draw attention. The tests for a word and for one lane read
     only what is at full strength, as the bracket gives them today.
   - *Graded tests.* One reading, and "one word" and "one lane" become
     matters of degree. This is the question carried to this item from
     1.8 §68.81.
3. *Dimming is the easy way to teach the place.* He deferred learning
   `.where` by gradient "unless you see a way to do that easily".
   Under a hard mask an event outside the area does not reach the cost,
   so the cost says nothing about whether it should have been inside.
   With a positive floor it reaches the cost in proportion to the
   floor, and each event's place factor has a gradient. What takes that
   gradient, and how it lives beside the discrete acts of `attend`, is
   not designed.
4. *The floor has a parameter.* `attentionFloor` would mean again what
   its name says. Whether the request's floor is the same number is not
   decided.

*What of §19.35 stands:* attention supplies the place and conceptual
space the request; attention's tests are taken on what comes back at
full strength (point 2, first way).

### 19.37 The gain field joins attention and expectation, and `attend` does not feed it (found for Alec, 2026-11-18)

Alec, 2026-11-18: "We had that dovetailing with expect in the gradient
calculation, but [I'm] not sure we integrated it with attend()".

**He is right on both counts** (read at `bfd2c606a3`).

*The design* (2026-10-19, the expectation specification): attention
and expectation are one gain field over a signal that is only ever
attenuated, `c = m * a * o`, with `o` the observed evidence, `m` the
attention mask and `a = (1 - g) + g s(r)` the factor of expectation:
what was predicted is dimmed toward the floor `1 - g`, and what
surprises is passed in full. The surprise is detached, so that a
predictor cannot lower its cost by dimming (the dark-room guard).

*It is built:* `Expectation.gain_field(observed, expected, strength,
attention, ...)`. It returns the dimmed content, the gain, and the
residual on two lanes, excess and omission.

*`attend` does not feed it.* Its four callers:

| Caller | What it passes as `attention` |
| --- | --- |
| `WordExpectation` (the word forecast) | which bank entries are valid |
| `SymbolField` | which streams are active |
| the symbol reader in `Expectation.py` | nothing (one) |
| `Meaning` (the close) | nothing (one) |

`recombine_percept` has the same argument and both its callers leave it
at one. So `m` is a yes or no for "this row exists" and never the place
`attend` chose. The place acts earlier and by a hard test: the look's
bracket decides which events are read at all (`read_percepts`,
`contained`), and the gain field sees what is left with `m` at one.

*Three more facts.*

- The forecast of a look, built today (1.8 §64, `LookForecast`), takes
  its excess and omission by a function of its own (`difference`),
  beside the gain field and not through it.
- In the content the whole gain is cut from the graph
  (`content = gain.detach() * observed`), the attention factor with
  it. As built, nothing can be taught through the gain, neither the
  forecast nor a mask.
- The gains are 0 in `data/model.xml` (`perceptExpectationGain`,
  `symbolExpectationGain`), so today `a` is one as well. *Not checked:*
  other configurations.

**What integrating it would be** (Claude's reading; it is §19.36 said
in his earlier formula). At a look, one gain field over the sentence's
percept tensor: `o` the events' activity, `m` the place `attend` chose
(one inside, `attentionFloor` outside) times the request (§19.32), and
`a` from the look's forecast. Its content is what the Pi matches
through; its excess and omission are the surprise `attend` reads,
which `LookForecast.difference` computes today.

**The point that needs his word when this is specified.** A mask
"learned by gradient" (§19.32) cannot be taught through the gain as it
is cut today. The guard he decided was on the surprise. To teach the
mask, the attention factor would stay in the graph while the factor of
expectation stays cut. This is the open point of §19.32 (which wires
take the gradient) in its exact form.

### 19.38 Two attentions, kept apart: the locus and the conceptual request (Alec, 2026-11-18; guidance from the literature)

Alec, 2026-11-18: "It helps to clarify the two types of human
attention; there is the locus of attention, and the top-down
conceptual attention. We are mixing these two a bit, and could
probably use guidance from the psychological literature."

**His own book first.** Chapter 16 of *The Whole Part* (the books
attached to the author's ORCID record) already separates them, and
adds a third:

- attention as an aperture that narrows awareness to a place, trained
  "mereologically" (areas of space, a sustained location);
- attention guided from below (what is intense, unexpected or valent)
  and from above, where the way down (visualization) works by
  inhibiting the wholes outside the scope, and is limited to concepts
  that already exist;
- attention to a referential level (the occurrence of a thought, or
  what it refers to).

**The literature** (from memory, by author and year; to be checked
against the sources before any of it enters a standing document).

| Finding | Source | What it says for the model |
| --- | --- | --- |
| Attention to a place and attention to a feature are distinct. Attention to a feature acts over the whole field, also outside the place attended | Treue and Martinez-Trujillo 1999; Saenz, Buracas and Boynton 2002 | The request is not confined to the look's `.where` |
| The two change gain independently and are controlled from different sources | Hayden and Gallant 2009; Moore and Armstrong 2003; Baldauf and Desimone 2014; Corbetta and Shulman 2002 | Two quantities, two teachers; `attend` is not asked to choose the request |
| Features are registered in parallel over the field; the locus is needed to bind them into an object | Treisman and Gelade 1980 | The request works at the first look, over everything; `interpret` happens only at the locus |
| What is wanted guides where the locus goes next, through one map of priority that also takes what stands out | Wolfe 1994; Itti and Koch 2001; Fecteau and Munoz 2006 | The request and the surprise are both signals to `attend`; neither is the place |
| A held template biases the competition among what is present | Desimone and Duncan 1995 | The request is a state held in conceptual space (already cited in `doc/Architecture.md`) |
| One field of attention over place and feature multiplies the input before normalization | Reynolds and Heeger 2009 | The gain field `m * a * o` of §19.37, with a normalization it does not have |
| Two separate numbers, a weight for which object and a bias for which category, multiplied | Bundesen 1990 | The same two factors, stated as a rate |
| What is unattended is attenuated, not blocked, and breaks through when it is expected or important | Treisman 1964 | The floor; and the tie of the floor to expectation |
| The locus has a size, and a wider one resolves less; its edge is a gradient | Eriksen and St. James 1986; LaBerge and Brown 1989 | `focus` and `unfocus` are a zoom; the place factor falls off and has no hard edge |
| What is selected is an object, and attention spreads within its bounds | Duncan 1984; Egly, Driver and Rafal 1994 | The selected set is tokens, not bytes (§19.34) |
| Orienting is pulled from below or sent from above, and what was selected before also counts | Posner 1980; Corbetta and Shulman 2002; Awh, Belopolsky and Theeuwes 2012 | Surprise, the request, and the lesson's history |
| In reading, the shape of the next word is taken from outside the locus to aim it, and attention moves a word at a time | McConkie and Rayner 1975; Reichle and others 1998 | A whole asked for ("letters, no spaces") aims the look before the word is known |

**Where we were mixing them** (Claude's reading).

1. §19.32 to §19.36 multiplied the place and the request into one
   activity for the Pi, as if the request were a second filter inside
   the look. The literature has the request acting over the whole
   sentence. Its first use is to say where the locus should go; its
   second is to dim, inside the locus, what was not asked for.
2. `attend` was being asked whether it should choose the request
   (§19.35). In the literature the two have different sources.
3. The third attention of Chapter 16, to a referential level, is in
   the model already under another name: `interpretBias` (1.8 §68.79),
   his "We should be able to treat a word as a word, not as what it
   represents". It is neither the locus nor the request.

**The three, in the model's names** (a reading; nothing is decided).

| Attention | In the model | Chosen by | Form |
| --- | --- | --- | --- |
| The locus | the look's `.where`; `focus` and `unfocus` as its zoom | `attend` | a place factor with `attentionFloor` outside it |
| The conceptual request | a demand over wholes or concepts, read down to percepts (§19.31, §19.33) | conceptual space: expectation and desire | a factor over kinds, acting on the whole sentence; limited to what is known |
| The referential level | `interpretBias`: the word, or what it stands for | a number now; experience later | a probability |

`attend` then reads a priority for each token (its surprise, and how
well it matches the request) and chooses the locus; the Pi matches
through the gain field of §19.37 at the locus; the grammar interprets
at the locus only.

### 19.39 One mechanism: the locus is where the request matches (Alec, 2026-11-18; in discussion)

Alec, 2026-11-18, on §19.38: "Unless locus of attention is also
top-down, the where also being driven by conceptual content? That
seems more simple, if we can do it."

**It can be done for the match, and it is a position the literature
holds.** §19.38 put the separation too firmly. What the literature
separates is asking for a place and asking for a kind. Biased
competition (Desimone and Duncan 1995) and Bundesen (1990) treat a
place as one more thing a bias from above can name, with no spotlight
of its own: the locus is whatever wins. In guided search (Wolfe 1994)
the locus is computed from the match of what is wanted. From memory,
as in §19.38.

**The mechanism exists in the source.** A demand over concepts has no
coordinates; it is given a place by the percepts present
(`cs_percept_attribution`: "Concept rows have no event coordinates;
only the supplied perception can locate a demand"), and
`_passback_scope_ps` already turns that into a strength for each part
(§19.33). The locus is then the tokens the request matches. No
coordinates are chosen by anything.

**What it would be.** One request from conceptual space, matched
against the sentence's percept tensor; the match is the activity the
Pi reads through (§19.34, §19.37); the locus is where that activity is
high. The bracket table and a chooser of places are not needed for
this.

**Four things the acts of `attend` do today that a match of content
does not give.**

1. *Moving on.* A request for words matches `red`, `cat` and `sleeps`
   alike. What was read has to give way. In the literature this is
   inhibition of return (Posner and Cohen 1984): what was just attended
   is dimmed for a time, and a serial scan falls out of one fixed
   priority. In the model it is the "record of having examined a
   particular place" he set aside as future work on 2026-11-18, and
   here it is needed: a token that was interpreted is dimmed (the
   `done` mark of a bracket today).
2. *Order.* Which of the three first. The tree is composed in the
   order the words are interpreted, so reading needs the earlier word
   to win. A content has no place, so this is a preference over place.
3. *Closer.* An unknown word is entered by its parts (`focus`,
   descend). By request that is "smaller wholes, inside this token",
   and "this token" is a place.
4. *One at a time.* The match is graded and several tokens are active;
   `interpret` wants one word. Either something picks a winner, which
   he has asked me not to propose, or the test is graded (the question
   already carried to this item).

**So the simple form needs one more thing: a place that can be asked
for as content.** Points 2 and 3 are requests about place. *Read, not
traced to the end:* the place of an event in the sentence is a pair of
integers today (`spans`); the code that `WhereRegistry` gives a percept
says which percept it is in perceptual space, not where it stands in
the sentence. A code for the position that a request can match is
what §19.32 called teaching the place "in the same way". Words of
place are item 1.6's.

**What `attend` becomes** (a reading). The chooser of the request: of
the kind asked for and of its grain (a word, its letters, a larger
whole), with the place following from the match. Its and, or and not
are already operations on content. It then chooses among kinds where
today it chooses among brackets.

**Size.** The loop of looks built in item 1.8, its lesson
(`AttentionLesson.ReadingTargets`) and its departures are written over
acts on brackets. This replaces them, so it is rule 32's size and not
a step inside item 1.8.

### 19.40 The first Pi ranges over one percept tensor and selects the where as it selects the what (Alec, 2026-11-18; in discussion)

Alec, 2026-11-18:

> If we let the first basic level pi operate over the entire unit
> hypercube of percepts it can select the what; it ought also be able
> to select the where, since concepts have that information embedded.
> So I'm hoping to remove the .where as an explicit lookup, and just
> let that pi range over some large perceptual tensor read into memory
> at the beginning of each sentence (or possibly two muxed what and
> where tensors)

**In plain terms.** Each percept of the sentence is one point: what it
is and where it stands. A row of the first Pi is a region of that
space (§18.14: a hypercone on the hypercube's parts and wholes). A
region that is wide along where and narrow along what is "the letter
c, anywhere". A region narrow along where is a place. A look is then a
region and not a test: its extent along where is the place, its width
is the zoom, and its soft edge is the dimming of §19.36.

**What exists** (read at `bfd2c606a3`).

- *The tensor.* The sentence's percepts are held once as indices and
  spans (§19.34), and a subspace's event is already what, where and
  when side by side: the content, then four rungs of where
  (`nWhere`, default 4), then when.
- *The where as a test.* `read_percepts` (`contained`),
  `PartSpace.part_memberships` (overlap of a part's span with the
  bracket, on the host), `narrowing_view` (`covered`) and
  `LookForecast` (`contained`) each compare integer spans.
- *A look's place in unit range.* `_passback_scope_where` is the look's
  interval divided by the length of the input, so the place is already
  an interval of 0 to 1 on one axis.

**Agreed as a direction (Claude), and it is the easy way to teach the
place** that he deferred in §19.32: a region's centre and width along
where are numbers of a row, and a soft edge gives them a gradient.

**Three conditions.**

1. *There are two wheres.* Where a thing stands in the sentence, and
   how its parts stand to one another. A concept holds the second (c
   before a before t) and, by rule 31, not the first: a generalization
   holds no place. Today the second is not a coordinate: a word's
   parts are matched as an ordered, adjacent run
   (`part_memberships`: "Groups preserve order, repetition and
   adjacency"). If the tensor holds only the place in the sentence, a
   row cannot tell `cat` from `act`; and with no arrangement at all,
   "red cat sleeps" holds every letter of `trade`. (People do this
   outside the locus: illusory conjunctions, Treisman and Schmidt
   1982, from memory.) One form that keeps it to one tensor: the look
   maps its place onto the unit interval, so that inside a look the
   where coordinate is relative to the look, and one row (c at the
   start, a in the middle, t at the end) matches the word wherever it
   stands. That would make "the unit hypercube" literal.
2. *The where is also an answer.* A word that is known is found by its
   run, and the place of the match is where it is. So for what is
   known, no place has to be asked for: the Pi over the whole sentence
   finds the known words at once, each with its place. A place has to
   be asked for only for what is not known, for the order of
   interpretation, and for one word at a time (§19.39).
3. *`.where` stays as a coordinate, and as what a reading carries.*
   The order of the words in the tree, the forecast of a look ("span
   left out"), and the places of what is held in STM all read it. What
   goes is the where as a test made on the host.

**Not known.**

- Whether the four rungs of where are a coordinate a region can bound
  smoothly, or a code in which an interval is not a region. Not
  checked.
- An extent is two numbers (a start and an end, or a centre and a
  size). If size is a coordinate, asking for words or for letters is a
  selection on it, and the zoom needs no act.
- The cost: rows of the Pi against events of a sentence. Not measured.

**What it touches.** Rule 32 and the first layer of this item, and the
percept pass of item 1.5. The Codex text of this item (§19.8) was
written before §19.31 to §19.40 and should not be carried until they
are settled.

### 19.41 Placed in item 1.6 (Alec, 2026-11-18; corrected by §19.42, and withdrawn by §19.44: the work is this item's)

Alec, 2026-11-18, on §19.40: "This is the 1.6 work I think."

*Read as* today's item 1.6: the number modality, the math corpus and
the felt world (1.8 §71). Until that morning the number 1.6 meant the
wiring work, which is now the first part of item 1.5; if he meant
that, this note is wrong and is to be corrected.

*Why it fits there* (Claude). Item 1.6's design
(`doc/plans/2026-11-18-item1-6-felt-math-world.md`) is where places
become coordinates on more than one axis (its §3.2), where one reader
serves every modality (§3.3), where a look over a scene is a box
(§3.7), and where a word of place has to be designed (§6). The things
§19.39 and §19.40 found missing are those: a place that can be asked
for as content, and a where the first Pi can bound.

*What moves.* §19.31 to §19.40 are item 1.6's: the request for a kind
of whole or for concepts, the mask on what is active, the percept
tensor read once for a sentence, the gain field fed by attention, the
locus as where the request matches, and the where as a coordinate of
the first Pi. They stay written here, where they were discussed; item
1.6's design is kept by another session and is not edited from this
one.

*What stays in this item* (Claude's reading; his placement of
2026-11-18 in §19.28 is not withdrawn): rule 32's Pi that finds the
spellings of a word, and hit and miss feeding its width. Rule 32's
first unspecified point, what a look's `.where` is when it is not the
stem's word intervals, goes with item 1.6.

*On §19.8.* The caution at the end of §19.40 falls away: this item's
text does not wait for §19.31 to §19.40. It waits, as before, only for
what of rule 32 remains in this item.

*The order.* Item 1.6 follows this item and comes before item 1.5. Its
slow tier is run once at its submission, and the architecture is to
change no more after it (1.8 §72), so this work has to be inside item
1.6's submission.

### 19.42 Placed in item 1.5 (withdrawn by §19.44); the filter from above and the selection from below (Alec, 2026-11-18)

Alec, 2026-11-18, correcting §19.41:

> Ah, yeah. The wiring work and single sentence percept banks, becomes
> 1.5. So this discussion about attention moves there also.
>
> I think we may already be doing what we need. A projection of the
> conceptual activation down to the basic percept pi would be the right
> filter, but it is not 1:1 (and I don't know how to extract the range
> spec from that). carrying the matching percepts under that pi to the
> conceptual activation would do the proper selection one by one, but
> that is computationally expensive.
>
> Thoughts?

**Placement.** §19.31 to §19.40 are item 1.5's, with the wiring and the
percept bank of one sentence. §19.41 read "1.6" as the number modality
and is withdrawn in that.

**The two routes, in the source** (read at `bfd2c606a3`).

- *From above.* `cs_percept_attribution` projects a demand over
  concepts down through the definitions of order zero to the basic
  percepts, sharing it over a concept's alternatives by their weights
  (§19.33).
- *From below.* The first look reads every word of the sentence in one
  pass and gives each its concepts (`FamiliarTier.open_sentence`,
  kept in `_sentence_open_fields`, one reading for each word with its
  place). The comparison of a content with conceptual space is one
  function (`ConceptualMatch.match`, 1.8 §70).

**Thoughts (Claude).**

1. *No range has to be extracted from the projection.* The projection
   gives kinds: which basic percepts are asked for. The range is read
   off the bank: each percept of the sentence has its place, and the
   places of those that match are the range. This is what
   `_passback_scope_ps` does, and it is condition 2 of §19.40: the
   where is an answer. If one range is wanted as numbers (to hand to a
   row of the Pi as a region), it is the centre and the spread of the
   matching activity along where, which needs no threshold. *The form
   is mine.*
2. *Not being 1:1 does no harm to a filter that only dims.* One
   percept serves several concepts, so the projection lets through
   more than was asked. Over-inclusion is the safe direction for
   something that lowers what is outside it and never removes
   (§19.36). It would not be safe as the selection itself.
3. *The selection from below is already paid for at the words.* The
   first look has carried every word up to its concepts once. To
   select among them is to compare the request with each word's
   reading, one `match` for each word, on tensors that exist. It is
   costly only where the first look made no reading: inside a word, or
   over several words taken as one.
4. *So the two are two stages, not two choices.* From above, cheap and
   over-inclusive, to dim and to say where to look; from below, exact,
   at the few places that survive. This is guidance and then binding
   (§19.38: Wolfe 1994; Treisman and Gelade 1980), and it is how the
   exact tests can stay exact: they are taken on the selection from
   below.

**What is and is not "already being done".** Both computations exist.
Neither is what a look selects by: the loop of looks selects by
brackets, the projection runs only in the old pass-back outside it,
and both build new tensors with loops on the host.

**One consequence of the placement, to be settled.** Item 1.5 is
written as wiring that keeps results equal, and it follows item 1.6,
after which "the architecture changes no more" and the slow tier has
run (1.8 §72). Reading the percept bank once and replacing the where
tests by coordinates can keep results equal. Selecting by a request
where brackets select today, and dimming where a bracket cuts, change
what the model does.

### 19.43 Item 1.5 may change the architecture (see §19.44, which moves the work here). The selection matrix from below (Alec, 2026-11-18)

Alec, 2026-11-18:

> 1.5 can change the arch, it will be the last round to do so.
>
> I think we can do it bottom up from percepts as a series of matmuls,
> as the architecture already does. At the level of the top-down
> filter, we have a mask of what we want (let's say the meta-concept).
> So the matrix that we want to compute is the selection matrix over
> percepts that light up that concept row.

**Decided.** Item 1.5 may change what the model does, and is the last
item that does. The question at the end of §19.42 is answered: the
selection by a request stays in item 1.5. 1.8 §72 said the
architecture changes no more after item 1.6; it is amended there.
*Not said:* whether the one overnight run of the slow tier moves from
item 1.6's submission to item 1.5's.

**What the selection matrix is** (Claude's reading). The way up is a
chain of matrices, percepts to basic percepts (Pi) to concepts (Sigma)
and on through the orders. A mask over the top, pulled back through
the same chain, gives for each percept of the sentence how much it
lights the wanted row. This is the backward product of the chain: the
derivative of the wanted row's activity with respect to each percept's
activity, times that activity. It uses the matrices of the way up and
no others.

*In the record already.* Rule 28 (§19.12) has thought read a row of
the Pi downward (`parts(c)`, "at every lower order", each with its
weight) and keeps the compressed rows readable by row and by column.
Rule 30 reads the first layer backward. The selection matrix is those
readings as one product, in place of a serial query.

**It answers both of his doubts of §19.42.**

- *Not 1:1.* The backward product goes through every path and weighs
  each by what is present, so it is not the projection of the weights
  alone.
- *One by one, and costly.* It selects one by one in one pass: its
  cost is that of the way up, once for a request, and not once for
  each candidate.

*A small case, run by me* (three tokens, `cat`, `cot`, `sat`; one row
of the Pi for the word `cat` as a product over its letters; `cat`
asked for):

| | `cat` | `cot` | `sat` |
| --- | ---: | ---: | ---: |
| Projection of the weights (letters of `cat` present) | 3 | 2 | 2 |
| Backward product times activity | 3 | 0 | 0 |

The projection lets `cot` and `sat` through with two letters of three.
The backward product keeps only the token that lights the row, because
the derivative of a product with respect to one letter is the product
of the others.

**Conditions.**

1. *It needs the Pi of this item.* The fold of today takes a minimum
   and keeps exact rails (`FieldLogic.power`), whose derivative is zero
   almost everywhere. With rows that are products it is as above. Item
   1.7 comes before item 1.5, so the order is right.
2. *It inherits the binding of the way up.* The small case binds
   letters by token. Over a whole sentence with no arrangement, the
   backward product of a row for `trade` on "red cat sleeps" is as
   wrong as the way up is (§19.40, condition 1).
3. *Two lanes.* A percept that lowers the wanted row, through a Not,
   comes back with the other sign. That is the against lane of the
   selection.
4. *How it is computed.* Either by the framework's backward pass over
   the look's graph, or by the transposes of the same sparse matrices
   with the local derivatives kept from the way up. *Default (mine):*
   the second, since it is wiring, needs no second backward pass inside
   the forward (2.7: one backward), and leaves open whether the mask is
   taught through it.

### 19.44 Decided: the attention and the sentence's percept bank are this item's; 1.6 proves, 1.5 compiles (Alec, 2026-11-18)

Alec, 2026-11-18:

> Ugh.
> Let's move this attention and sentence percept bank work to 1.7,
> then, with the rest of the pi-not-sigma work.
> 1.6 is the proving ground for what it is doing
> 1.5 is the compilation to make it fast

**Decided.** §19.31 to §19.43 are item 1.7's, with the Pi, Not, Sigma.
The placements of §19.41 (item 1.6) and §19.42 (item 1.5) are
withdrawn.

| Item | What it is |
| --- | --- |
| 1.7 | the Pi, Not, Sigma; the looks, the request, and the percept bank of one sentence |
| 1.6 | "the proving ground for what it is doing": the number modality, the math corpus and the felt world |
| 1.5 | "the compilation to make it fast": computation by wiring, then the lookups, the folds and the percept pass |

*Claude's reading, his to correct.* His sentence of §19.43, "1.5 can
change the arch, it will be the last round to do so", was said to make
room for this work in item 1.5. He has not taken it back in words. With
the work here, item 1.5 is described again as making fast what exists,
so rule 3 of 1.8 §72 is read as it was first written: the architecture
changes no more after item 1.6, and the slow tier is run once at item
1.6's submission. The question at the end of §19.43's reply falls away.

**What it means for this item.**

- The Codex text of §19.8 was written before §19.31 to §19.43. It is
  not to be carried as it stands. (§19.41 had lifted this caution; it
  is back.)
- This item now holds, beside rules 1 to 32: the percept bank of a
  sentence read once (§19.34, §19.40); the first Pi matching it through
  what is active, with no `materialize()` on that path (§19.34); a
  look as a place and a request (§19.31 to §19.33, §19.39); attention
  as the filter that dims, fed into the gain field with expectation
  (§19.36, §19.37); the selection matrix from below (§19.43); and the
  where as a coordinate of the first Pi (§19.40).
- All of it is in discussion. Before a text can be written it has to
  be made into rules, and these need his word:
  1. which wires take the selection's gradient (§19.32, §19.37);
  2. whether a look removes what was not asked for or only dims it
     (§19.33);
  3. whether the tests for one word and one lane read what is at full
     strength, or become graded (§19.36; and the "Felix" case carried
     from 1.8 §68.81);
  4. the two wheres, a place in the sentence and the arrangement of a
     thing's parts (§19.40);
  5. what moves the look on, and in what order (§19.39);
  6. who changes the request inside a sentence (§19.35).

### 19.45 `attend`, tentatively: ask by concepts, pull back, read what the way up returns (Alec, 2026-11-18)

Alec, 2026-11-18:

> The attend() method is what is changing: we ask for percepts by
> selecting some set of concepts, doing the pull (back?), and examining
> the set of returned word concepts for something that we might want
> to interpret(). But that set of "returned concepts" is just conceptual
> activations from the forward pass after the pullback.
>
> Will that work tentatively? If so, let's write that to the 1.7 space

**Yes, tentatively** (Claude). "Pullback" is the word: it is what the
backward product of §19.43 is called.

**The steps of one `attend`.**

1. *Ask.* A set of concepts is selected: a mask over concept rows, on
   two lanes.
2. *Pull back.* The mask is carried down through the matrices of the
   way up (§19.43). Each percept of the sentence's bank gets a share:
   how much it lights what was asked.
3. *Go up again.* The way up is run on the bank with that share as its
   activity (§19.34). What it gives is the returned set: the concepts
   that what was selected supports, each at the token where it is.
4. *Examine.* The word concepts in the returned set are the candidates
   for `interpret`.

**On "red cat sleeps".** Asked for words: the pullback keeps the three
runs of letters and drops the spaces; the way up returns `red`, `cat`
and `sleeps`, each at its token. Asked for the concept of `cat`: the
pullback keeps the letters of the second token alone, and the way up
returns `cat`. Asked for `dog`: the pullback is zero, because a row
that is a product has no share where a part is missing, and nothing is
returned. So what comes back is what was asked for and is present, not
merely what was asked for.

**Why it holds together.**

- It adds no structure. Steps 2 and 3 are the matrices of the way up,
  read down and then up, over a bank that is read once.
- It tests presence. A request cannot confirm itself (the `dog` case).
- A near miss comes back weakly. With graded rows, `caat` under a
  request for `cat` returns `cat` at less than one, which is rule 32's
  Pi finding the spellings.
- The acts of today have a place in it: `focus` is a narrower request,
  `unfocus` a wider one, and the and, or and not of the field are the
  same operations on the request's two lanes.

**What it needs, and my defaults** (each *mine*, each his to change).

1. *Rows that are products* (§19.43, condition 1): this item's Pi.
2. *The binding of the way up* (condition 2): the bank keeps the
   stem's tokens, as the first look reads them today, until rule 32
   says otherwise.
3. *A request must be able to name a whole.* A word not yet known
   lights no word concept, so a request by concepts never returns it.
   Asked for by its whole ("letters and no spaces", WholeSpace's fixed
   word row, §19.31), its token comes back with the whole lit and no
   word, which is the "neither" that leads to a new name today.
4. *Where the request comes from.* *Default:* from desire, the channel
   the record gives to selection in comprehension (the expectation
   specification, §4.7; `prime_desire`), and not from a choice among
   all concept rows, which would be a menu as large as the vocabulary.
   Expectation is not the request: it dims what it predicted (§19.37).
5. *Moving on, and order.* *Default:* a token that was interpreted is
   lowered in the bank's activity (the `done` mark of today), and among
   the candidates the earliest in the sentence is taken first.
6. *The tests.* `interpret` is offered at a returned token that is one
   word with one lane, as now. Exact or graded is the question carried
   from 1.8 §68.81.

**What it replaces, when it is made into rules.** The bracket table
and its acts on brackets (`Attention.narrow_step`), the look's read by
a containment test (`SentenceLooks.perceive`, `read_percepts`), and the
lesson written over those acts (`AttentionLesson.ReadingTargets`). The
grammar's option at a look (`interpret_at_look`, `interpretBias`) and
the forecast of a look stay.

**Not checked.** None of this is run. Before rules are written it is
simulated on the lesson's sentences with a first layer of products
(my rule: simulate before specifying).

### 19.46 Item 1.7 after 2026-11-18: the looks, as rules (Claude; tentative, a consolidation of §19.31 to §19.45)

Alec, 2026-11-18: "1.7 spec." This section states the discussion of
§19.31 to §19.45 once, as rules that continue §19.21's numbering. They
are *tentative*: none is built on until the prototype of §19.47 has
run and the questions below have his word. §19.21 stands for rules 1
to 32, and §19.8 is rewritten from both when they are settled. His
words are in the sections named; what is marked *default* or *reading*
is Claude's.

**F. The looks.**

33. *The percept bank of a sentence* (§19.34, §19.40). At the start of
    a sentence its percepts are read once into one bank: for each
    event its row (the what), its span (the where), its time, the
    token it belongs to, and its activity on two lanes. Parts and
    wholes are events on one axis, as `read_percepts` joins them
    today. Nothing on a look's path calls `materialize()`, and nothing
    on it reads a span on the host. *Default:* the tokens are the
    stem's, as the first look reads them, until rule 32's Pi makes
    them.
    *Added, 2026-11-18 (Claude, from the measurement in
    `doc/plans/2026-11-14-item1-5.md`):* the lookup of the sentence's
    percepts is made once, at this one place, and the Pi is read only
    on the rows it names, not on every row of the dictionary. How the
    lookup is computed (the trie on the host today, a table on the
    device later) is item 1.5's and changes no result.
34. *A look is an activity over the bank* (§19.32, §19.34). It takes
    the place of the bracket's interval and of the containment test.
    An event's activity in a look is its own, times the place factor
    (rule 39), times its share of the request (rule 36). What is not
    active is absent on both lanes and is never evidence against
    (his rule for `non`, 2026-11-12).
35. *The request* (§19.31, §19.33, §19.45). A mask over concept rows
    on two lanes. It may name a whole: WholeSpace's canonical
    properties are rows it can name, the fixed word row among them.
    *Default:* it comes from desire (`prime_desire`; the frozen
    concept "reading", on from the start, asks for words), and is not
    a choice among all rows.
36. *The pullback* (§19.43). The request is carried down through the
    matrices of the way up, and each event of the bank gets a share:
    the derivative of the asked rows' activity with respect to the
    event's activity, times that activity. It uses rule 2's Pi, whose
    rows are products, and the Not and the Sigma above it. *Default:*
    computed by the transposes of the same sparse matrices with the
    local derivatives kept from the way up, with no second backward
    pass inside the forward.
37. *The returned set* (§19.45). The way up is run again on the bank
    with the share as its activity. What it gives, each concept at its
    token, is the returned set. The word concepts in it are the
    candidates for `interpret`.
38. *`attend`* (§19.45). One `attend` is: ask, pull back, go up,
    examine. The acts on brackets go. A narrower request or place is
    `focus`, a wider one `unfocus`, and the and, or and not of the
    field are operations on the request's two lanes. *Defaults:* a
    token that was interpreted is lowered in the bank's activity (the
    `done` mark of today), and among the candidates the earliest in
    the sentence is taken first.
39. *Attention dims; it does not cut* (§19.36). The place factor is
    one inside the look's area and `attentionFloor` outside it. The
    factor is applied once to a token's reading, on both lanes, and
    not inside the product of a row's parts.
40. *One gain field* (§19.37). A look's activity is the `attention`
    of `Expectation.gain_field`, and the look's forecast (1.8 §62.3)
    is its `expected`. Its excess and omission are the surprise
    `attend` reads; `LookForecast.difference` goes.
41. *The where is a coordinate* (§19.40). An event's span is a
    coordinate of the bank that a row of the first Pi can bound; the
    tests of spans made on the host go (`contained`,
    `part_memberships`' overlap, `covered`). `.where` stays on every
    reading. *Default:* how a thing's parts stand to one another stays
    what it is today, an ordered and adjacent run, computed on the
    device; the form in which a look maps its place onto the unit
    interval is the alternative (question 4).

**What stays.** The grammar's option at a look and `interpretBias`
(1.8 §60.1, rule 5b); the forecast of a look from what lies outside it
(1.8 §62.3); `interpret` at one word with one lane; rule 32's Pi over
spellings.

**What goes.** The bracket table and its acts
(`Attention.narrow_step`, `narrowing_view`); the look's read of one
bracket (`SentenceLooks.perceive`); the old pass-back
(`_passback_scope_ps`), which the pullback replaces; and the lesson
written over acts on brackets
(`AttentionLesson.ReadingTargets`; question 7).
`cs_percept_attribution` keeps its other readers, the reverse of the
field among them, until rule 30's way back takes their place.

**Certificates, in outline** (numbered when the rules are settled).

- The bank is read once: on a look's path no `materialize()` and no
  host read of a span, counted on the lesson's sentences.
- A request for a word's concept returns it at its token; for a word
  that is absent, nothing; for words, every run of letters. Beside
  each, the projection of the weights alone, which lets near words
  through (the small case of §19.43).
- A word not yet known comes back under a request for the word whole
  with the whole lit and no word concept, and is given a name.
- A token outside the look's area is present at the floor on both
  lanes and is never read as against.
- A fresh model with `interpretBias` at 1 reads every word of the
  lesson's sentences once, in order.
- On a hard mask (floor at zero) the surprise equals 1.8's.
- The host reads and the seconds of the sixteen sentences, beside
  1.8's.

**The order of building.** After the steps of §19.21, since rule 36
needs rule 2's Pi and rule 30's way back: the bank; the pullback and
the returned set, in tests first (§19.47); `attend` and the gain
field; then the removals.

**For Alec, before these are rules.** Each with what I would take.

1. *Which wires take the selection's gradient* (§19.32, §19.37).
   *Recommended:* the asking side only (desire, and `attend`'s own
   map); the pullback is cut from the first layer's weights, so that
   perception keeps its one teacher.
2. *Remove or dim* (§19.33). *Recommended:* dim, with
   `attentionFloor`, as he described attention.
3. *The tests for one word and one lane* (§19.36; 1.8 §68.81). Two
   parts. (a) Attention's dimming: *recommended* that the tests read
   what is at full strength, so dimming never makes a look "several
   words". (b) A learned complement, the "Felix" case he asked to be
   addressed: *recommended* an offer in proportion to how little the
   two lanes overlap (1.8 §68.81, the second choice), the form being
   his to set.
4. *The two wheres* (§19.40). *Recommended:* arrangement stays the
   ordered adjacent run in this item; the place relative to the look
   comes with rule 32's Pi.
5. *What moves the look on, and in what order* (§19.39).
   *Recommended:* rule 38's defaults.
6. *Who changes the request inside a sentence* (§19.35).
   *Recommended:* nothing in this item beyond the standing desire;
   thought's operators may write desire later.
7. *What teaches reading now.* The lesson of 1.8 teaches the fewest
   acts on brackets. With the request from a standing desire and the
   order by default there is nothing left for it to teach.
   *Recommended:* it goes, and `attend`'s zero-started map stays for
   what experience adds later.

### 19.47 Codex text: a prototype of the looks, in tests only (follows the last text of item 1.8)

Written on 2026-11-18. It builds nothing of the model. It is what
1.8 §68.84 names as the work that follows it (1.8 §75, point 4).
*Approved by Alec, 2026-11-18:* "Yes, carry both". *Amended later the
same day, after his answers of §19.48 and before Codex had begun it:*
the two domains of the pullback, the dimming by the asking row, a
request for a place, the fallback, and a trial of reading in order;
and after §19.49, the meta concept as the first request and the most
active word concept as what is taken.

~~~
Item 1.7, a prototype in tests only. Read §19.43, §19.45, §19.46,
§19.48 and §19.49 of doc/plans/2026-11-14-sigma-pi-layers.md, and §75 of
doc/plans/2026-11-11-item1-8-last-architecture-round.md, which holds
here: a question is not a stop; build what you recommend, mark it
provisional, go on. Work in /Users/arogers/github/WikiOracle/basicmodel
on main. Local commits only; never push main, never force-push. Change
no file of bin/ and no existing test. It decides nothing and is not
wired into the model. Where §19.48 and §19.49 differ from §19.46,
they stand.

1. Under test/, build a small first layer whose rows are products over
   a sentence's percepts, as rule 2 of §19.21 gives the Pi, with a
   Sigma above it. The sentence's percepts are held once (rule 33):
   for each, its row in the what codebook and its place as
   WhereEncoding gives it, the constant sinusoid. The stem's tokens
   are its readings.
2. On it, the steps of §19.45. A request over concept rows is pulled
   back through the Pi onto two domains, the what codebook and the
   where sinusoid, and the activity of each active percept is the
   intersection of its match in the two (§19.48, answer 2). Do the
   pullback by the transposes of the same matrices and, beside it, by
   the framework's backward pass, to show they agree. Then the way up
   again on that activity, and the returned set (rule 37).
3. Run it on the lesson's sixteen sentences with five requests: the
   meta concept that words are included in, as one row over the word
   concepts (the proof of concept of §19.49); a word's concept; the
   fixed word whole; a word that is absent; and a row that is narrow
   on where, a place. Report, for each, what is returned and at which
   token, beside the projection of the weights alone. From each
   returned set take the most active word concept, as interpret will
   (§19.49, answer 3): report whether it is the word meant, and where
   two are equally active.
4. Report the cost in operations beside one pass of the way up, and
   what the binding by token does and does not prevent: a sentence
   that holds every letter of a word it does not hold, and two words
   of the same letters in another order.
5. The dimming of rule 39 as §19.48 has it: by the asking row's radius
   and magnitude, with no fixed floor. Report what the returned set
   holds for the other tokens as the radius narrows and widens, when
   the factor is applied once to a token's reading and when it is
   applied inside the product.
6. The where sinusoid as a coordinate: how sharply a region around one
   place takes in its neighbours on the long rung and on the short
   one, and whether the short rung's repeats let in places far away.
7. The fallback of §19.48, beside the returned sets of 3: a .where
   given outright, and from that extent the largest word, or failing
   a word the largest concept.
8. If the rest is done: a small trial of §19.48, answers 1 and 5. With
   the gradient on the request alone, the meta concept asked for and
   a place beside it, can the request be taught to make the words of
   a sentence the most active one at a time and in order? Report what
   was taught, on how many sentences, and what it found.

Return it as §20 of doc/plans/2026-11-14-sigma-pi-layers.md: the files,
each result as a table, what surprised you, and what you would change
in rules 33 to 42. Name the test files you ran.
~~~

### 19.48 Alec's answers on the looks (2026-11-18), and what they change in rules 33 to 42

Alec answered the questions of §19.46 and the readings put to him with
them. His words are quoted whole; what follows each is Claude's
reading unless marked otherwise.

**1. Which wires take the selection's gradient.** "Those wires are just
the specification of the concepts that create the pullback".

- The gradient lands on the request: on which concepts are asked for,
  and on what produces that asking. This is what §19.46 recommended.

**2. Remove or dim.**

> dim it: but since attend is using the PI, the radius of the hypercone
> (and the magnitude) should determine that within the mereological
> field. since we allow the PI to specify both content and position, we
> need it to project onto two domains (the what codebook and a where
> domain that is just the constant sinusoid), and then take the
> intersection of those two across the active percepts. That will give
> us a set of the sentence which has been conceptually specified. If
> the output of attention is onto all concepts, then that wiring will
> be learned.
>
> Again, this is a prototype, we can always fall back to a discrete
> approximation if it doesn't work: select .where explicitly via
> network and select the largest word (falling back to largest
> concept) from that extent.

- *Dim, and the Pi says by how much.* What lies outside is lowered by
  the hypercone of the row that asks: its radius and its magnitude.
  The default of §19.46, a separate floor (`attentionFloor`), is
  withdrawn.
- *Two domains.* A request, pulled back through the Pi, is projected
  onto the what codebook and onto the where domain. The where domain
  is the constant sinusoid: `WhereEncoding` (`bin/Spaces.py`), the
  fixed two-rung code of a place, sines and cosines of its address,
  with nothing learned in it. Only a place's start is in that code;
  its end is not.
- *The intersection.* For each active percept of the sentence, its
  match in the what domain and its match in the where domain are
  intersected. What is left is the part of the sentence that the
  concepts specified.
- *The output of `attend` is over all concepts.* So the request is
  what `attend` gives, and its wiring is learned. The default of
  §19.46, that the request comes from desire and not from `attend`, is
  withdrawn. This also answers question 6 as Claude meant it (below).
- *The fallback, if the prototype does not work:* a network gives the
  `.where` outright, and from that extent the largest word is taken
  (§19.30), or failing a word the largest concept.

**3. The tests for one word and one lane, and the "Felix" case.** Not
answered. It stays open. His answer to 2 bears on it: where the cone
dims by degree, a test that counts any value above zero reads every
word of the sentence at every look.

**4. The two wheres.** "I don't think we need this: see 2".

- There is one where, the place on the sinusoid, and the Pi bounds it.
  Condition 1 of §19.40 is withdrawn as a requirement. Whether a row
  then tells `cat` from `act` is what the prototype measures (§19.47,
  point 4), and is not argued further here.

**5. What moves the look on, and in what order.** "presumably we want
to find words in order before doing any grammar on them. this can be
taught, and should be a test that gates grammar tests".

- Reading the words in order is taught, not built in. The default of
  §19.46, the earliest candidate first as a rule, is withdrawn.
- A test that the words of a sentence are found in order gates the
  grammar tests, as the assessment's first rung gates its others.
- *Not said:* whether every word is found before any is composed.
  Today each word is composed as it arrives (1.8 §60.1, rule 6).

**6. Who changes the request inside a sentence.** "what?" The question
was badly put. In one sentence `attend` is called several times, and
each call asks for some concepts; the question was what decides, at
each call, which concepts are asked for. His answer to 2 settles it as
Claude reads it: `attend` itself, whose output is over all concepts.

**7. Whether the reading lesson goes.** "see 5". It stays, and what it
teaches is finding the words in order. `AttentionLesson.ReadingTargets`
is rewritten for requests in place of acts on brackets, and is not
removed.

**The readings.** "Slow run: good. Part C: good. Codex's limit: OK.
interpretBias: OK." So: the one overnight slow run is at item 1.6's
submission (1.8 §72 as first written); part C of item 1.8 ended at
§66; under 1.8 §75 Codex does not overturn a recorded decision of his
in the source; and `interpretBias` is the probability of the call,
with the chooser deciding the rest.

**A word that is looked at and not interpreted.**

> Ah. "Is the conceptual activation that is created by the call to
> attend() something that should be treated as a word, and if not,
> what happens to the sentence". We can embed the word in the tree as a
> literal with the understanding that it should have been quoted "meow
> said the cat" might have a sign as the first word instead of a
> symbol. But since that wont play grammatically, it probably has to
> take the place of a clause in the sentence. We should allow it into
> the sentence only if there is a grammatical rule to take care of it
> (like sign()), otherwise it will not be a WFF.

- *Rule 42 (new, tentative).* A word left uninterpreted may enter the
  tree as a literal, a sign and not a symbol, as if it had been
  quoted. It stands where a clause would. It enters only where the
  grammar has a rule that takes it, such as `sign`; where none does,
  the sentence is not well formed.
- *What exists.* The rule is there: `SignLayer` (`bin/Interpret.py`,
  `rule_name = 'sign'`, one operand, "Keep the lexical carrier for
  generation"), declared in `data/complete.grammar`, and the kept
  close already sets apart "a leaf kept as a sign"
  (`FamiliarTier.commit_read_words`). *Read, not traced:* what does
  not exist is the way to it from a look. A word reaches the grammar
  today only when `interpret` is called on it, so a word left
  uninterpreted is in no tree; and `sign` takes a word, where his
  literal takes the place of a clause.

**The rules of §19.46 as they now stand.**

| Rule | Change |
| --- | --- |
| 33, the bank | none here; one lookup for a sentence, and the Pi read only on the rows it names (added in rule 33) |
| 34, a look is an activity | the activity is the intersection of the what match and the where match (answer 2) |
| 35, the request | it is the output of `attend`, over all concepts, and is learned (answers 1, 2, 6); it may still name a whole |
| 36, the pullback | it projects onto two domains, the what codebook and the where sinusoid (answer 2) |
| 37, the returned set | none |
| 38, `attend` | the order is taught and tested; no built-in order (answer 5) |
| 39, dimming | by the asking row's radius and magnitude; no separate floor (answer 2) |
| 40, one gain field | none |
| 41, the where | one where, the sinusoid; no second where for arrangement (answer 4) |
| 42, a literal | new (above) |

**Certificates added.** The words of a sentence are found in order,
and this gates the grammar tests (answer 5). A literal enters only
under a rule that takes it (rule 42).

**Still open with him.** Question 3. And whether his answer to 6 is as
read. *Both answered the same day: §19.49.*

### 19.49 Alec's answers to questions 3 and 6 (2026-11-18)

**3. The tests for one word and one lane.** "we want to select the most
active word-concept to put on the stack during interpret(), so it must
be exact."

- *What `interpret` takes.* Of the word concepts in the returned set
  (rule 37), the most active one is put on the stack. It is a
  selection, with no draw and no degree: exact.
- *What it withdraws.* The graded offer that §19.46 recommended for
  the "Felix" case (1.8 §68.81, the second choice) is not taken.
  `interpretBias` stays what it is, the probability that the call is
  made at all (1.8 §60.1, rule 5b).
- *The dimmed field* (§19.48, answer 2) is then no difficulty for the
  test: every word of the sentence may be present to some degree, and
  the one that is most active is taken.
- *The "Felix" case, as Claude reads it; his to correct.* In this
  item a concept is one signed number (rule 3, the Sigma). A reading
  of for 1 and against 0.0002 is an activity just under one. It is
  still the most active word concept at its place, so it is
  interpreted. The count of lanes above zero, which left the word
  unread in item 1.8, is no longer what is tested. This is how the
  decision carried from 1.8 §64.7 ("Keep this baseline as long as we
  address it later") is addressed.
- *Left to the rules, with Claude's defaults:* where the returned set
  holds no word concept, nothing is interpreted; and where two are
  equally active, the one found first is taken.

**6. Who decides what each call of `attend` asks for.** "yes, attend
will apply the top-down mask on concepts that gets pulled back.
Eventually this will be top-down conceptual attention, as proof of
concept it could work by pulling back the meta-concept to isolate
words in the percept codebook."

- *Confirmed:* the request is `attend`'s own output, a mask over the
  concepts, and it is what is pulled back (§19.48, answers 1 and 2).
- *The proof of concept.* The request is the meta concept that words
  are included in (§19.29). Pulled back, it isolates the words among
  the percepts of the sentence. Later the mask is whatever conceptual
  attention asks for.
- *With answer 5 of §19.48,* as Claude reads the three together: the
  meta concept alone returns every word of the sentence equally
  active. The where part of the request (the second domain of answer
  2) is what makes one of them the most active, and reading in order
  is `attend` learning to move that place along.

**Rules 37 and 38, as they now stand.** The returned set is the way up
on the pulled-back activity; `interpret` takes its most active word
concept, exactly. One `attend` is: give the mask over concepts, pull
it back onto the what codebook and the where sinusoid, intersect, go
up, and hand the returned set to the grammar.

**Nothing of §19.46's seven questions is open.** The rules can be
written when the prototype (§19.47) has returned.

### 19.50 Review of §20, the prototype of the looks (Claude, 2026-11-19)

**Result.** Accepted as a prototype. It changes no file of `bin/` and
no existing test: three files are new under `test/`
(`looks_prototype.py`, `looks_prototype_report.py`,
`test_item17_looks_prototype.py`). Its nine tests pass here (3.1 s),
and its report prints the tables of §20.2, §20.3 and §20.4 as
returned (compared line by line; the tables of §20.5 to §20.8 were
read and not compared line by line).

**What it shows.** The look as Alec described it (§19.48, §19.49)
works on the lesson's sixteen sentences. A request for the meta concept
"word" with a place returns the word at that place in 16 of 16, and a
word that is absent returns nothing in 16 of 16, on three conditions:
the Sigma is passed by its weights, a place is read once at a token's
start, and of equally active words the largest is taken. A look costs
about 2.3 ways up, once for a request and not once for each candidate.
Reading in order was learned from the request alone: one number, the
step from the end of the word read to the next place, taught by the
gradient through the pullback, and after 20 steps all sixteen sentences
were read in order.

**Checked by Claude.**

1. *A membership of exactly one passes no gradient through `PiLayer2`*
   (§20.2). Confirmed in this environment (torch 2.15.0): for
   `exp(3 log clamp(m, eps, 1))` the gradient at `m = 1` is 0 and at
   `m = 0.999` it is 2.994; with the floor alone it is 3.0 at `m = 1`.
   The clamp is in the source (`bin/Layers.py`, `PiLayer2`, built by a
   kernel option and by a fold). Whether it costs today's training
   anything was not checked.
2. *A product row is a test of containment* (§20.4). Confirmed, and it
   depends on what the bank holds. The prototype's bank holds bytes and
   a word's row asks for its letters. The model's own code of a word is
   the pairs of `#word#` and the lengths 1 to n
   (`WordIdentity.base_atoms`). A word is "wholly present at another
   token" when every part its row asks for is in that token:

   | A word's row asks for | On the lesson's sixteen sentences | In a dictionary |
   | --- | --- | --- |
   | its letters (the prototype) | 8 cases, in 6 of 16 (`is` in `This`, `here` in `There`) | not run |
   | its pairs and lengths (the model's atoms) | 0 cases | 938 of 3,000 sampled words (31%) wholly contain another word (`the` in `there`, `my` in `mystifyingly`) |
   | the same, with the next length absent | 0 cases | none; 64 of 197,014 words (0.03%) share pairs and length with another word (`acara`, `araca`) |

   | Asked | Token | letters | pairs and lengths | with the next length absent |
   | --- | --- | --- | --- | --- |
   | `Felix` | `xileF` | present | absent | absent |
   | `is` | `This` | present | absent | absent |
   | `here` | `there` | present | absent | absent |
   | `the` | `there` | present | present | absent |
   | `cat` | `caat` | present | present | absent |
   | `cat` | `act` | present | absent | absent |

   This is a count over sets of atoms (the scratch script
   `word_rows_check.py`, not kept in the repository), not a run of the
   prototype's products. The dictionary is the system's word list, 2
   to 12 letters, lower case.
3. *The store has a row for each word form and for each atom, and the
   lookup names them* (`WordIdentity.word_rows`, `atoms`, `parts`): a
   word's row, its atoms' rows, and its witnessed nested wholes.

**Where Claude reads the return differently.** Two of its five
surprises come from the prototype's bank of bytes, and rule 33's bank
is not that. Rule 33's bank holds what the sentence's one lookup names:
for a token, the row of its form where the store knows the form, the
rows of its atoms, and its place, which is its start.

- *The contained word* (§20.4; §20.9, point 3). A known word's concept
  asks for its form's row, and the lookup has already made that an
  equality: `is` is not the form at `This`. The atoms are for a form
  that is not known, rule 32's reading of spellings, and there a row is
  a containment test with the numbers above.
- *The word cut by its own place* (§20.5; §20.9, point 4). An event of
  the bank has one place, its start. The where match is read there,
  which is §20.5's "read at the token", the reading that took the word
  at the place at every concentration.
- So the two changes §20.10 proposes for rule 41, the return of the
  ordered run or letters at their places within the token, are not
  taken. Order inside a word is in the word's own code, the pairs of
  `#word#`.

**Where Claude agrees.** The Sigma is passed by its weights and not by
its slope (§20.9, point 1): a saturated Sigma has almost no slope, so
the concept that is most present would return least. Alec's words for
the pullback are "a series of matmuls" and "the selection matrix"
(§19.43), which are the matrices. A share becomes the look's activity
capped at one (rule 34). Of equals the largest is taken, his own idea
(§19.30), then the one found first.

**Not checked.** The counts of multiply-adds (§20.4), the profile of
the sinusoid (§20.6) and the trial of reading in order (§20.8) beyond
their tests passing.

### 19.51 The looks, as rules after the prototype (Claude, 2026-11-19; from §19.46, §19.48, §19.49 and §20)

These replace the wording of rules 33 to 41 in §19.46 and add rule 42.
What Alec decided is given with its section. What is marked *mine* is
Claude's and is his to change; under 1.8 §75 it does not hold the
item's text.

**F. The looks.**

33. *The percept bank of a sentence* (Alec, §19.34, §19.40). At a
    sentence's start one lookup reads its percepts into one bank. For
    each token the bank holds the row of its form, where the store
    knows the form; the rows of its atoms; its place, which is its
    start, with that place's code on the constant sinusoid
    (`WhereEncoding`, as the registry builds it, both rungs); its
    time; and its activity on two lanes. Nothing on a look's path
    calls `materialize()` and nothing on it reads a span on the host.
    The Pi is read only on the rows the lookup names. How the lookup
    is computed is item 1.5's and changes no result. *Mine
    (§19.50):* the events are the lookup's rows and not bytes; the
    tokens are the stem's until rule 32's Pi makes them.
34. *A look is an activity over the bank* (Alec, §19.48, answer 2). An
    event's activity in a look is its own, times its match in the
    what domain, times its match in the where domain: the
    intersection of the two across the active percepts. What is not
    active is absent on both lanes and is never evidence against (his
    rule for `non`, 2026-11-12). *Mine (§20.10):* the intersection is
    the product; a what match is the event's share of the request,
    capped at one; an event that two asked rows share is not counted
    twice.
35. *The request* (Alec, §19.48, answers 1, 2 and 6; §19.49, answer
    6). It is the output of `attend`: a mask over all concepts, on two
    lanes, and it is learned. It may name a whole. The gradient of a
    look's selection lands on the request, on which concepts are asked
    for and on what produces that asking. As proof of concept the
    request is the meta concept that words are included in, with a
    place. *Mine:* the first layer's weights take none of that
    gradient, so that perception keeps its one teacher.
36. *The pullback* (Alec, §19.43; §19.48, answer 2). The request is
    carried down through the matrices of the way up onto two domains,
    the what codebook and the where sinusoid, and each event gets a
    share. *Mine (§20.9, §20.10, §19.50):* a Pi is passed by the
    derivative of its product and a Sigma by its weights, not by its
    slope. It is computed by the transposes of the same sparse
    matrices with the Pi's local derivatives kept from the way up,
    with no second backward pass inside the forward. A membership of
    exactly one passes its derivative; in the source `PiLayer2`'s
    clamp does not, and that is corrected with a test. Where a token
    has several copies of one row its membership is the greatest of
    them, and each copy at the greatest takes the whole derivative.
37. *The returned set* (Alec, §19.45; §19.49, answer 3). The way up is
    run again on the bank with the look's activity. What it gives,
    each concept at its token, is the returned set. `interpret` takes
    its most active word concept, exactly: a selection, with no draw
    and no degree. *Mine:* among equals the largest (his idea,
    §19.30), then the one found first; where the set holds no word
    concept, nothing is interpreted.
38. *`attend`* (Alec, §19.45; §19.48, answers 5 and 7; §19.49). One
    `attend` is: give the mask over concepts, pull it back onto the
    two domains, intersect, go up, and hand the returned set to the
    grammar. The acts on brackets go. No order is built in: reading
    the words of a sentence in order is taught, and a test of it gates
    the grammar tests. The reading lesson stays and is rewritten for
    requests in place of acts on brackets. *Mine:* the first thing
    taught is where the next place is, from where the last word ended
    (§20.8); a narrower request or place is `focus` and a wider one
    `unfocus`, and the and, or and not of the field act on the
    request's two lanes; a token that was interpreted is lowered in
    the bank's activity, the `done` mark of today; each word is
    composed as it arrives, as today (1.8 §60.1, rule 6), which his
    answer 5 does not say.
39. *Attention dims; it does not cut* (Alec, §19.36; §19.48, answer 2).
    What lies away from the asked place is lowered by the hypercone of
    the row that asks, by its radius and its magnitude, within the
    mereological field. There is no separate floor. *Mine (§20.5):*
    the weights of a place row set its radius; the row's value at its
    own place falls as the radius narrows, and the Sigma's weight on
    the row restores it; the where match is read at an event's place
    and applied once to the token's reading, on both lanes, not inside
    the product of a row's parts. The key `attentionFloor` is not read
    by a look.
40. *One gain field* (*mine*, from what §19.37 found for him: the
    gain field joins attention and expectation, and `attend` does not
    feed it). A look's activity is the
    `attention` of `Expectation.gain_field`, and the look's forecast
    (1.8 §62.3) is its `expected`. Its excess and omission are the
    surprise `attend` reads; `LookForecast.difference` goes.
41. *One where* (Alec, §19.48, answer 4). The where is the place on
    the constant sinusoid, and a row of the first Pi bounds it. There
    is no second where for how a thing's parts stand to one another.
    The tests of spans made on the host go (`contained`,
    `part_memberships`' overlap, `covered`). `.where` stays on every
    reading. *Mine (§19.50, §20.6):* order inside a word is carried by
    the word's own code, the pairs of `#word#`. With the registry's
    capacity the short rung separates the words of a sentence and the
    long rung removes the short rung's repeats 256 bytes away; a
    capacity of 64 gives both rungs one period, and is not used.
42. *A literal* (Alec, §19.48). A word left uninterpreted may enter the
    tree as a literal, a sign and not a symbol, as if it had been
    quoted. It stands where a clause would. It enters only where the
    grammar has a rule that takes it, such as `sign` (`SignLayer`);
    where none does, the sentence is not well formed.

**Rule 32, a note from the prototype** (*mine*). A form the store does
not know is read by its atoms, and a row over atoms tests containment
(§19.50): `the` is wholly present at `there`. Where the longer form is
known, its own row is the larger of the two and is taken (rule 37).
Where it is not known, the shorter word would be read in its place.
The lengths 1 to n say "at least n"; saying "exactly n" needs the next
length's absence, and absence is not evidence (rule 34). This is left
to rule 32 with the numbers of §19.50 and is not settled here.

**The fallback** (Alec, §19.48), if the build does not work: a network
gives the `.where` outright, and from that extent the largest word is
taken, or failing a word the largest concept. §20.7 shows it agrees
with the returned set wherever one word is at the place. It needs no
request, so nothing in it can be taught to ask.

**What stays and what goes** are as in §19.46, with two changes: the
reading lesson (`AttentionLesson.ReadingTargets`) is rewritten and not
removed, and `attentionFloor` is no longer read by a look.

**Certificates** are those of §19.46 and §19.48, and from the
prototype: a word that is absent returns nothing; a known word is not
taken at the token of a longer known word that holds its letters; a
membership of exactly one passes its derivative through the Pi; a
request for the meta concept with a place returns the word at the
place on each of the lesson's sentences.

**For Alec, three points; each is built as recommended unless he says
otherwise.**

1. *The Sigma in the pullback.* *Recommended:* by its weights. The
   other choice, by its slope, returned nothing for a word that is
   fully present.
2. *What the bank holds.* *Recommended:* the rows the lookup names, a
   token's form and its atoms. The other choice, bytes, makes a word's
   row a bag of letters.
3. *Among equally active words.* *Recommended:* the largest, then the
   first found.

**What comes next.** The item's text for Codex (§19.8) is rewritten
from §19.21 (rules 1 to 32) and from this section, as one hand-off.

### 19.52 Alec on §19.51: no derivative in a look, the codes, and ties (2026-11-19)

Alec, of the three points of §19.51 and the two questions put with
them:

> 1 in the pullback (?), passing by weights is fine, but I worry that
> what you have done is rely on the learning algorithm. Can you point
> me to an article online that shows the principle you are exercising,
> hopefully Wikipedia? Also, can you check if the derivative issue is
> due to preexisting stimulation, and possibly work around that by
> doing the inverse, which we had been running anyway?
>
> 2 OK. Does the mereological space of codes still get produced?
>
> 3 if we have equally active words, then we have not identified the
> correct word. Please an example.
>
> Questions 1 please explain in more detail what it means to have a
> read with no sentences 2 wait until discussion finishes

**1. The derivative was Claude's, and it is not needed.** The share
"activity times the derivative of the asked rows" is Claude's reading
in §19.43. His words there were "a series of matmuls" and "the
selection matrix over percepts that light up that concept row". The
source carries a request down today by an inverse that shares a
concept's demand over its entries by their weights
(`cs_reverse_presence`, `cs_percept_attribution`, read by
`_passback_scope_ps`), with no derivative; §19.46 had that replaced by
the pullback, which was the wrong way round.

*His first check: is the derivative issue due to preexisting
stimulation? Yes.* The slope of a Sigma is what a little more input
would add. Where the first pass has already lit the concept, it adds
almost nothing:

| The row's value | The concept, `tanh(3 z)` | Passed down by the slope | Passed down by the weight |
| ---: | ---: | ---: | ---: |
| 0 (at rest) | 0.000 | 3.000 | 3.0 |
| 0.1 | 0.291 | 2.745 | 3.0 |
| 0.5 | 0.905 | 0.542 | 3.0 |
| 1 (wholly present) | 0.995 | 0.030 | 3.0 |
| 2 (two words in one token) | 1.000 | 0.00007 | 3.0 |

At rest the slope is the weight, exactly. "By the weights" is the
derivative taken before anything was stimulated. The zero that
`PiLayer2` gives at a membership of exactly one (§19.50) is the same
kind: full stimulation stands at the clamp's ceiling.

*His second: work around it by the inverse. Yes.* Run by Claude on the
lesson's sixteen sentences and three more ("Fe li x is here.", "xileF
is here.", "The cat is there."), each asked for each of fifteen words
and for the meta concept "word": 304 requests. A request is carried
down three ways, then the way up is run again, and what counts is
whether the returned set is exactly the asked words that are present
(a word concept at 0.01 or more).

| A word's row reads | By the slope | By the weights (a derivative) | By the rows' entries (the way back, no derivative) |
| --- | ---: | ---: | ---: |
| its letters (the prototype) | 230 of 304 | 256 of 304 | 203 of 304 |
| its own form and its letters (rule 2) | 230 of 304 | 304 of 304 | 304 of 304 |

"By the rows' entries" is: read downward the entries the asked rows
have weight on (rule 28's `parts(c)`), and keep what of them is
active. With rule 2's row, which reads its own entry and its parts, it
gives the same sets with the same values as the derivative by the
weights in 304 of 304, and no derivative is taken. With rows of
letters alone it is worse (a word's letters light other words made of
them), which is one more reason the row reads its own form. The
script is `way_back_check.py` in the session's scratch folder and is
not kept in the repository.

*The principle, and where it is written.* Recall through one matrix
upward and its transpose downward, with no learning step in the
recall, is the bidirectional associative memory
(https://en.wikipedia.org/wiki/Bidirectional_associative_memory). A
top-down template compared with the bottom-up input, with a
"vigilance" that sets how close the match must be, is adaptive
resonance theory
(https://en.wikipedia.org/wiki/Adaptive_resonance_theory); its
vigilance is close to rule 32's width of a row's cone. What Claude had
done, the derivative of an output by the input, is the gradient kind
of saliency map (https://en.wikipedia.org/wiki/Saliency_map), which is
the learning algorithm's backward pass turned on the input. The three
pages were read on 2026-11-19 for these statements.

**Rule 36, restated on this** (*Alec's suggestion, confirmed by the
check; his to confirm*). The request is carried down by the way back:
through a Sigma to the rows it has for entries on, through a Pi to the
entries a row has weight on, onto the what codebook and the where
sinusoid. No derivative is taken and nothing of the learning
algorithm runs in a look. An event's what match is its own activity
where an asked row reads it, and nothing where none does. The
existing readers (`cs_reverse_presence`, `cs_percept_attribution`) are
the place to start from, not something to remove. What §19.51 said of
the Pi's local derivatives, of the copies of a letter and of a
membership at one leaves rule 36; the clamp of `PiLayer2` stays a
point for the source, since training passes through it.

**2. The mereological space of codes.** It is still produced; nothing
in rules 33 to 42 touches what makes it. The store gives each form a
code that is the union of its parts' codes (`WordIdentity`: the pairs
of `#word#`, three coordinates of 64 each, and the lengths 1 to n, one
coordinate each); the sentence's parts and wholes are read once
(`read_percepts`); the concept codes are form and meaning
(`MereologicalCodes`). The bank is that mereology read once, each
event with its code.

*What the prototype did not do.* It made a table of bytes of its own
and never read these codes. So it says nothing yet about a row of the
Pi as a cone in the space of codes.

*Measured in the codes themselves* (20,000 words of the system's word
list; `code_containment_check.py`, scratch): no two words have the
same code; 194 of 1,500 words (13%) hold the whole code of another,
shorter word. `the` lies inside `there` and `cat` inside `caat`; `is`
does not lie inside `this`, nor `cat` inside `act`, nor `here` inside
`there`. So in the codes equality tells words apart and containment
gives the neighbours, which is what a part and whole space should do.

**3. Equally active words: his point is taken.** "Then the one found
first" (§19.51, rule 37) is withdrawn; it picked a word where none was
identified. Examples:

- *No place asked.* The meta concept "word" alone, on "This is
  Felix.", returns `This`, `is` and `Felix`, each at 0.995. Nothing is
  identified; the request has to say where.
- *The same word twice.* A request for `is` in a sentence with two
  returns both equally, for the same reason.
- *The prototype's ties* (`is` at `This`, 4 of 16 with a place) come
  from rows of letters and are gone when a row reads its own form: in
  the 304 requests above a single word asked for never returned two.
- *Two forms at one place.* His example of §19.30: a look at "ice
  cream" where the model holds `ice cream`, `ice` and `cream`. `ice
  cream` and `ice` begin at the same place and both are wholly present.

**Rule 37, restated on this** (*Claude's reading of his remark; his to
confirm*). `interpret` takes the most active word concept of the
returned set. Where two or more are equally the most active, no word
has been identified: nothing is interpreted, and `attend` asks again.
The one case with a rule is his own (§19.30): of forms that begin at
the same place the largest is the word.

**His two questions.** The read with no sentence is explained to him
in the conversation (1.8 §66.12 holds it); his word on keeping it is
still to come. The item's text for Codex (§19.8) is not written until
this discussion is finished: "wait until discussion finishes".

### 19.53 Alec on §19.52: the salience map selects the level, the where selects the word; parthood of forms; a read that gives no WFF (2026-11-19)

Alec:

> The salience map is great for attention, it does seem to be selecting
> the relevant words, but if it selects multiple we want it to narrow,
> perhaps by running the pi or entire network with the chosen word
> (probably better done randomly) as rightly or wrongly chosen... but
> the salience map shouldn't be selecting everything, it just selects
> word-level, and the choice of word is done by the .where selection
>
> If in the mereology "is" is not a part of "this", we have to fix it
> (or are you saying no parthood can be determined because you are
> looking at the whole space and they have the same code?) Please add a
> test point to ensure the validity of the structure as composed.
>
> Equally active words. See what I said above about .where being used
> to select the right word horizontally, while the salience selects the
> right level.
>
> If the read does not produce a WFF, that's an error to propagate and
> we can try again. But the best way around this is to start with
> minimal sentences, where the error drives exactly the difference
> between a shorter successful parse and the erroneous parse.
>
> Questions 1. See above 2. The interpret had been finding the object
> corresponding to an object. Maybe we can Test word recognition in a
> bunch of garbage letters, and set the desired output to be just the
> word? I feel your solution is a bit intrusive.

**The term.** The map that a request leaves over the percepts of a
sentence is the *salience map*. *Claude's reading:* he names the map,
however it is computed, and §19.52 computes it by the way back with no
derivative. If he means the gradient kind by that name, rule 36 goes
back to it and he will say.

**1. Two selections** (Alec). The salience map selects the level: the
word level. It does not pick the word. The `.where` selects the word,
"horizontally". So the meta concept "word" returning the three words
of "This is Felix." is the map doing its part, and the where does the
rest. In rule 35's terms the request has a what part, which gives the
level, and a where part, which gives the word.

**2. When several are still selected, narrow** (Alec): "perhaps by
running the pi or entire network with the chosen word (probably better
done randomly) as rightly or wrongly chosen". *Claude's reading:* one
of the selected words is taken by a draw, the Pi or the whole network
is run with it, and the outcome shows whether it was rightly chosen.

- *In the source already.* A draw of this kind is a departure. Each
  explore path departs once, in attention or in another walk, and is
  judged by the cost of its whole path (`explorePaths`,
  `SentenceCredit.WALKS`); the grammar's decision at a look is
  recorded, explored and credited that way (`interpret_at_look`).
- *The gap.* Today the draw is over acts on brackets and over the
  grammar's option. Here it is over the words that the map and the
  where left selected.

**Rule 37, restated** (*Claude's reading; his to confirm*). The
salience map gives the words of the sentence at the word level, and of
forms that begin at one place the largest (§19.30). The where selects
one of them and `interpret` takes it. Where several remain selected,
one is taken by a draw, and the draw is credited by the outcome of its
path as a departure is. §19.52's "nothing is interpreted" is
withdrawn.

**3. A read that gives no WFF** (Alec). It is an error; the error is
propagated, and the sentence may be tried again. The way around it is
the order of teaching: begin with minimal sentences, so that a parse
that fails differs from a shorter parse that succeeds by one step, and
the error drives exactly that step. *Claude's reading:* the lessons
that teach reading begin with the shortest sentences and grow a step at
a time. It bears on rule 38 (the reading lesson, and the test of
reading in order that gates the grammar tests) and on rule 42 (a
literal with no rule to take it is not well formed). It also bears on
item 1.6's corpus, which is another session's plan and is not edited
from here. *Not traced:* what the source books today when a sentence
does not reduce; the forced reductions of the closing make every
sentence end as a row.

**4. Parthood of forms: `is` in `this`.** His question first: no, the
two do not have the same code, and no two of 20,000 words did
(§19.52). Parthood is determined, and it comes out partial. Measured
on the store itself (`RadixLayer` with `WordIdentity`, 29 words, one
suffix lesson for `cats`; scratch script `structure_check.py`):

| The test | Result |
| --- | --- |
| Every part the store records for a form lies wholly inside the form's code | 233 of 233 |
| A known form found inside a known form lies wholly inside it | 4 of 36 pairs |
| A form that is not inside another reads as wholly inside it | 1: `cat` in `caat` |

| Part | Whole | How much of the part is found in the whole |
| --- | --- | ---: |
| `is` | `this` | 0.82 |
| `his` | `this` | 0.86 |
| `at` | `cat` | 0.89 |
| `here` | `there` | 0.88 |
| `the` | `there` | 1.00 |
| `cat` | `cats` | 1.00 (recorded by the suffix lesson) |

*Why.* A form's code is the pairs of `#word#`, with the two ends
marked, and the lengths 1 to n. `is` begins a word and the `is` in
`this` does not, so one of its three pairs is missing there. The store
records a form inside a form only as a stem and one marker witnessed
in a suffix lesson (`observe_nested`; its comment: "There is no
vocabulary-wide substring search").

*So it is to be fixed, by his word.* The item 1.5 plan noted this on
2026-11-18 as what today's codes do not give "if it is ever wanted:
exact parthood between forms", when keeping the codes was confirmed.
It is now wanted. Three ways, measured by Claude on the system's word
list (197,015 forms of 2 to 12 letters; 1,897,751 pairs of a form
inside a form, 9.6 for each form; scratch script
`parthood_options_check.py`):

| The code of a form | A form inside reads wholly inside | A form not inside reads wholly inside | Coordinates on, of 96 | Forms that share a code |
| --- | ---: | ---: | ---: | ---: |
| today's | 5% | 0 of 270,182 | 32 | 36 |
| today's, joined with the code of every known form found inside it | 100% | 100 of 270,421 | 44 | 224 |
| its pairs with no end marks, and its lengths | 100% | 43 of 270,121 | 28 | 161 |
| the overlap tree (his, in the item 1.5 plan) | 100%, exactly | none | a row for every form inside a form, 9.4 rows for each word, and a wider code | none |

With 256 coordinates for the pairs in place of 64, the second way has
8 false parts and 68 shared codes, and the third 7 and 113.

*Recommended:* the second. It is the store's own mechanism: `form()`
already joins a form's code with the codes of its recorded parts
(`nested`), and what changes is what is recorded, every known form
found inside and not only a stem and its marker. It keeps the end
marks, which keep words apart. *Not designed:* how a form learned later
is added to the longer forms that already hold it (the store rekeys
every form when `nested` changes today), and how many collision mints
it needs at this scale.

**The test point he asked for** (a certificate of this item, numbered
when the rules are settled): *the structure as composed is valid.* On
the lesson's words, after a suffix lesson and after a collision mint:

- (a) every part the store records for a form, each atom and each
  recorded form inside it, lies wholly inside the form's code (today
  233 of 233);
- (b) every known form found inside a known form lies wholly inside
  it, read by the one match (`ConceptualMatch.match`): `is` in `this`
  reads 1 (today 0.82; 4 of 36 pairs);
- (c) the count of forms that read as wholly inside a form they are not
  inside is reported, with `cat` in `caat` named (today 1); what a
  near spelling should read is rule 32's.

Part (a) holds today and (b) fails today; (b) passes when the fix is
built.

**5. The read with no sentence** (1.8 §66.12). *Claude's reading of
his words:* the repair is not wanted as built. `interpret` finds the
object that corresponds to a word; the repair calls it from the
preparation of the input, outside a sentence, which is a second copy
of the decision the sentence's loop makes. His proposal in its place:
test word recognition among garbage letters, with the word alone as
the desired output. As Claude reads it, that is an ordinary case of
input and output run as a sentence, so nothing reads outside a
sentence. It is also the proof of concept of the looks (§19.49, answer
6: the meta concept pulled back isolates the words), so it is a
certificate of this item: *a word among garbage letters is returned,
and it alone is said.*

*Proposed, for his word:* the three commits of the repair are taken
back (`26666553c0`, `0a7f25a805`, `64a8be7fc1`); the two callers, the
fast word evaluator and the standalone kernel, read inside a sentence
run or lose those cases; his test is added. *Not checked:* whether the
evaluator can read inside a sentence run at small cost, and whether
item 1.8 as built passes his test where the garbage is separate runs
of letters.

### 19.54 Alec on §19.53: the complete part graph, no end marks, a space character (2026-11-19)

Alec:

> Parthood of forms, the code of a form: Lets explore the powerset of
> parts of a whole, over the existing parts. So the complete part
> graph, and the most efficient storage of that graph.
>
> Salience and where: I guess we had salience selecting where also;
> perhaps we need more training, and we need to specify where in the
> conceptual tree (which will restrict the salience beyond just
> "words").
>
> A read with no WFF: we will probably have to return to this
>
> The read with no sentence: same as a read with no WFF, your solution
> sounds better: maybe generate the one-word read by bisecting the
> failed sentence for training parts, expecting the word thus isolated.
>
> Q 1. lets discuss the powerset of parts instead of just pairs, see
> above. Do drop the end marks, add a space character so that the
> saliency may can select against if from within parts. 2. see above

**Decided** (Alec). The end marks go: a form's code no longer marks its
first and last letter. A space character is added, so that the
salience map can select against it from within parts. The code of a
form is to be explored over its parts of every size, not its pairs
alone.

**The reading of "the powerset of parts"** (*Claude's*). A whole's
code is the set of the existing parts inside it, of every size. That
set is one point of the powerset over the existing parts; the sum of
two wholes is the union of their sets, and one is a part of the other
when its set lies inside the other's. The complete part graph is the
relation "is inside" among the existing parts. A part is taken here as
a run of adjacent characters; a collection of letters that are not
adjacent (`t` and `s` of `this`) is not taken as a part.

**What the space does** (*Claude's reading*). The space is a part like
a letter. A part that holds a space has the space among its parts, and
a request selects against it by a Not entry on the space (rule 3).
What is left lies within words, and the largest of those at a place is
the word (§19.30). The same character lets a form of two words exist
(`ice cream`, his example), which a request that does not select
against the space can take. What makes something a word is then the
level a request asks for, and no longer part of the form's code: `is`
has one code wherever it stands, inside `this` or alone. *In the source
today:* the space has a property row of its own (`whitespace`, among
WholeSpace's canonical properties), and the store is built so that no
stored part crosses a space (`RadixLayer(word_bounded=True)`).

**The complete part graph, measured** (Claude; the system's word list,
197,014 forms of 2 to 12 letters, no end marks; scratch script
`part_graph_check.py`). Two choices of which parts exist:

| | E: only the parts that exist (the list's own forms, and single letters) | C: every run of letters of every form |
| --- | ---: | ---: |
| Nodes | 197,040 | 1,493,626 (7.6 for each form) |
| Every pair "is inside", written out | 1,881,807 (9.55 for each form) | not counted |
| Only the pairs "is immediately inside" | 522,429 (2.65 for each form) | 2,987,200 (two for each node) |
| Parts in a form's code | 17.7 | 42.8 |

- *The most compact exact storage of the graph is its immediate
  edges.* A partial order is fully given by the pairs with nothing
  between them (its transitive reduction, the Hasse diagram:
  https://en.wikipedia.org/wiki/Transitive_reduction). Everything
  inside a form is found again by walking down. In E that is 522,429
  edges in place of 1,881,807 pairs, 3.6 times fewer.
- *In C the immediate edges are exactly two for a node:* the part
  without its last letter and the part without its first. This is his
  rule of 2026-11-18 (the item 1.5 plan: a whole stored as the overlap
  of its two parts). A whole is fixed by its two parts in order: of
  1,493,600 pairs of parts none has two wholes. So in C a sentence's
  parts can be built from the letters up, one table lookup for each
  length, at every place at once.
- *E has no such build.* 524 forms hold no known form at all, and a
  form is not made from its known parts (`this` is not made of `his`
  and `hi`). In E the parts of a sentence are found by the lookup at
  each start place, as today, and the graph says only which form is
  inside which.

**The code, from the graph.** With one coordinate for each existing
part, a form's code is the set of the parts inside it, itself
included. Then parthood is exact both ways: a form is inside another
exactly when its set is inside the other's, and no two forms share a
code. The set is small (17.7 parts in E) and the space of coordinates
is as large as the number of parts, so the code is stored as a short
list and not as a wide row. Folded into a row of fixed width, three
coordinates for each part (E):

| Width | A form not inside reads as wholly inside | Forms that share a code |
| ---: | ---: | ---: |
| 96 (today's) | 20 of 273,750 | 154 |
| 256 | 1 of 273,750 | 12 |
| 512 | 0 of 273,750 | 2 |
| 1,024 | 0 of 273,750 | 0 |

A form inside another reads as wholly inside in every case, at every
width, by construction.

**His part map, added the same day** (Alec, while this was being
written): "we might be able to build the part map as "all pairs of a
word, no mention of the ends, and add a three or four character part to
disambiguate two words like "banana" and "bananana"." Measured by
Claude on the same list with `bananana` added (197,015 forms; counted
on sets of parts, nothing folded; scratch script
`pairs_part_map_check.py`):

| | Count |
| --- | ---: |
| Forms that have the same pairs as another form | 1,536, in 746 groups |
| The same, with the length kept beside the pairs | 224, in 112 groups |
| Pairs of such forms told apart by a run of 3 letters | 790 (94%) |
| by a run of 4 letters | 46 (5%) |
| by a run of 5 letters | 5 (1%) |
| A form not inside another, all of whose pairs are in it | 16 of 517,334 (`dog` in `odontology`) |
| The same, when its runs of 3 letters must be in it too | 0 of 517,334 |

- *It works.* A run of three or four letters tells apart 99 in 100 of
  the forms that pairs leave the same.
- *His own example needs five.* `banana` and `bananana` have the same
  pairs, the same runs of three and the same runs of four; the run
  `nanan` is the first that only one of them has. The length also
  tells them apart.
- *One condition, so that "is inside" stays true.* A longer part that
  is added becomes an existing part, and every form that holds that
  run carries it. If only the two words that needed it carried it, a
  form with that part would stop reading as inside a longer form that
  holds it.
- *Where it sits among the choices above.* It is E with the fewest
  parts: all pairs, and a longer part only where two forms would be
  the same. The known forms inside a form need not be coordinates.

**For discussion, with what Claude would take.**

1. *Which parts exist.* *Recommended:* his part map: all pairs, and a
   longer part where two forms would be the same, each carried by
   every form that holds it. It is E at its smallest. C is the one to take if the parts of a
   sentence are to be built from the letters up by lookups on the
   device, which is item 1.5's question (the radix trie as a tensor).
2. *How the code is held.* *Recommended:* the short list of parts as
   the exact code, and a folded row of 512 for where a fixed width is
   needed.
3. *The lengths 1 to n* of today's code. With a coordinate of its own
   for each form they are no longer needed to tell forms apart. Not
   looked at further.

**Salience and where** (Alec). The salience was to select the where as
well (§19.48, answer 2: the Pi specifies both content and position);
that may need more training. And the request is to say *where in the
conceptual tree*, which restricts the salience beyond "words".
*Claude's reading:* a request names a node of the tree of concepts, at
any order, and the more particular the node, the fewer words the map
selects; the meta concept "word" is the broadest such node and the
proof of concept only. §19.53's two selections stand as the simple
case, not as a rule that the map never selects the where.

**A read with no WFF** (Alec): "we will probably have to return to
this". Open.

**The read with no sentence** (Alec). It is the same matter as a read
with no WFF. Of what §19.53, point 5, proposed: "your solution sounds
better". And a way to make the one-word reads: bisect the failed
sentence into parts for training, and expect the word thus isolated.
*Claude's reading:* the repair is to be taken back as proposed, and a
sentence that fails is halved until a word stands alone, each piece a
read whose expected output is what it holds. Since he returns to the
read with no WFF later, nothing is changed in the source now; this
goes with the item's text, unless he says to take the repair back
sooner.

### 19.55 Alec: matches over all locations by a matrix product, one matrix for every character length (2026-11-19)

Alec: "the idea is that we do matmul to find matches over all
locations, with a mat for every character length. That will give us
the set of matches for all parts, what will uniquely determine what
word is present."

**In plain terms** (*Claude's reading*). A character is one of a few
kinds (here 27: the letters and the space). A window of k characters
of the sentence is k blocks, each with a one at its character. A part
of k characters is a row of the same shape. Their product counts the
characters that agree, and the part is at that place when the count is
k. One product for a length gives every part of that length at every
place of the sentence at once.

**Run by Claude** (the system's word list, 197,014 forms of 2 to 12
letters; 100 sentences of 20 words, about 200 characters; scratch
script `matmul_match_check.py`). Two forms of it:

| | V1: every existing form is a row of the matrix of its length | V2: his part map, then a product from parts to words |
| --- | --- | --- |
| Rows | 197,014 in 11 matrices, at most 32,295 in one | 617 pairs and 367 longer parts (320 of 3 letters, 43 of 4, 4 of 5) |
| What it finds | every form at every place: 4,249 matches in 20 sentences, none different from a search of the text | the parts at every place |
| The word at a place | matched by its own row, 400 of 400 | the parts matched inside the word give exactly that word, 400 of 400; no two forms share their parts |
| Time for a sentence | 4.4 ms on Metal, 12.9 ms on the host | 19.8 ms on the host, as written (loops over words and a sparse product; not tuned, and not a fair number) |

- *His claim holds on this list.* With every pair and a longer part
  where two forms would be the same, the set of parts matched inside
  a word determines the word: none of the 197,014 forms shares its
  set with another.
- *One gap.* A pair that is no known part leaves no trace. The run
  `felixx` matched the parts of `felix` and nothing else, so it read
  as `felix`. For the set to determine the word, the run must also be
  matched at every place, or its length must agree. `xqzcatpl`, `zzz`
  and `thisis` gave no word.
- *"Inside a word"* was bounded here by the spaces, which is what the
  space character of §19.54 is for.
- *Cost beside the lookup of today.* The lookups measured for the item
  1.5 plan took 0.2 to 0.4 ms for a sentence with the model on Metal.
  V1 is some ten times that at this size and grows with the number of
  forms. V2's first product is small whatever the size of the
  dictionary; what grows is the product from parts to words, which is
  the first Pi (a word is the row whose parts are exactly those
  matched).

**Where it stands.** It is a way to compute rule 33's bank on the
device with products alone, and V2 is the part map of §19.54 put to
work. Not decided: which of the two forms, and whether it is built in
this item or with item 1.5's lookup.

### 19.56 Alec: pairs, too, only where needed (2026-11-19)

Alec, of Claude's "all pairs, plus a longer part only where needed":
"in fact we should also do pairs only when needed."

So the part map starts from the letters, and a part of any length,
pairs included, is added only where two forms would otherwise have the
same parts. Measured by Claude on the same list (197,015 forms, with
`bananana`; scratch script `minimal_part_map_check.py`). The parts are
chosen greedily: a group of forms with the same parts so far is split
by the shortest run that divides it most evenly, and a part once added
is carried by every form that holds it. This is an estimate and not
the least possible set.

| | Count |
| --- | ---: |
| Forms that have the same letters as another form | 140,320, in 35,979 groups (the largest of 63) |
| Parts added so that no two forms are the same | 737: 392 pairs, 297 of 3 letters, 43 of 4, 5 of 5 |
| Pairs needed, of the 617 that occur | 392 |
| Parts in a form's code (its letters and its added parts) | 15.9 |
| Forms that still share their parts | 0 |
| A form inside another reads as inside | 45 of 45 in the sample |
| A form not inside another reads as inside | 14 of 517,334 |

Against §19.54's map (617 pairs and 367 longer parts, 984 in all) it
is a quarter fewer parts, and parthood reads as well: `is` has the part
`is`, `this` has `hi`, `is`, `th` and `thi`, so `is` is wholly inside
`this`; `cat` has `at` and `ca`, `act` has `ac` and `ct`. `banana` and
`bananana` differ by `nanan`, as before.

*What it changes in §19.55.* The matrices for each length hold the
letters and these 737 parts in place of every pair, so the first
products are smaller still. Not run again with this map.

### 19.57 Alec: test the part map on Metal; "words" presses against the space; reconstruction (2026-11-19)

Alec:

> So if I understand correctly, the proposed "part powerset" is more
> efficient? Lets test that out, at least it runs on MPS and gives us
> the full set of parts on wires, as we had hoped.
>
> Please do make sure that selecting "words" in the concept tree gives
> negative pressure to the " " characters.
>
> On reconstruction, if we reconstruct a word poorly in virtue of not
> being able to combine parts at the right locations, will the
> inaccurate reconstruction create a differentiating word? Regardless,
> reconstruction will have to proceed from the specific concept, which
> will need to associate the part arrays (what) with matching location
> arrays (where).

**1. The test on Metal** (Claude; scratch script
`part_map_mps_test.py`). The part map of §19.56: the 26 letters and
736 added parts, 762 in all (392 pairs, 297 of 3 letters, 43 of 4, 4
of 5), for 197,014 forms; a form holds 15.9 parts. A sentence is 20
words, 201 characters. Stage 1 gives the parts at every place, one
product for each length. Stage 2 gives the words: the parts matched
inside each run of letters, in one product against the parts of every
form; a word is the row whose parts are exactly those.

| | On Metal | On the host |
| --- | ---: | ---: |
| Words found, and the right ones | 500 of 500 | 500 of 500 |
| Stage 1, the parts at every place | 0.50 ms | 0.44 ms |
| Stage 2, the words, against all 197,014 forms | 5.92 ms | 8.00 ms |
| Both | 6.25 ms | 8.45 ms |

One sentence gives an array of 201 places by 762 parts with 369 parts
at places, on the device.

*Is it more efficient?* In what is stored, yes: 762 parts in place of
a code of 96 coordinates for each form and a trie. In getting the
parts onto wires, it is of the same order as today's lookup (0.2 to
0.4 ms for a sentence with the model on Metal, the item 1.5 plan), and
it is products alone. In finding which form the parts make, not as
run here: one product against every form is 5.9 ms and grows with the
dictionary. That step need not be made against every form (rule 33:
the Pi is read on the rows that can be meant); how it is narrowed on
the device is not designed.

**2. "Words" presses against the space** (Alec). A request that
selects the node "words" in the concept tree gives the space a
negative entry: a Not entry on the space (rule 3). *Added to rule 35.*
*Certificate:* with a form of two words known (`ice cream`), a request
for "words" lowers the part that holds the space below the two words
inside it, and without that request the larger form is the one that
fits (§19.30); on the lesson's sentences no part selected by "words"
holds a space. *Not simulated:* the run above bounded a word by the
spaces outright.

**3. A word put back together wrongly** (his question). Run by Claude
with the same part map (3,000 words of four letters or more; scratch
script `poor_reconstruction_check.py`). A wrong string is read as the
set of parts found in it:

| The wrong string | Strings | Read as the same word | As another word | As no word |
| --- | ---: | ---: | ---: | ---: |
| two neighbouring letters exchanged | 22,609 | 0.1% | 0.3% | 99.6% |
| one letter left out | 25,609 | 0.5% | 3.7% | 95.8% |
| one letter said twice | 25,609 | 6.0% | 0.1% | 93.9% |
| a pair of letters said twice | 22,267 | 11.3% | 0.1% | 88.6% |
| the word's own pairs joined in another order, same length | 7,390 | 2.3% | 6.5% | 91.2% |

- *Mostly it makes no word,* which is seen as an error.
- *Sometimes it makes another word* (`pilage` with a letter said
  twice is `pillage`; `peopler` with a letter left out is `people`).
  That is a differentiating word in his sense, as Claude reads it.
- *Sometimes it still reads as the word that was meant,* and the part
  map cannot tell: `pimpleprooof` reads as `pimpleproof`. The part map
  tells known forms from one another, not a known form from every
  string. Where the length is compared as well, none of the strings
  with a letter or a pair added or left out reads as the same word;
  what is left is 0.1% of the exchanges and 2.3% of the other orders.

**4. Reconstruction proceeds from the specific concept** (Alec). The
concept associates its part arrays (what) with matching location
arrays (where). *Claude's reading:* a form keeps, beside the parts it
holds, the place of each within it. The read needs only the set; the
way back needs the places. Stage 1 gives them: each part is matched
at a place, and within a word those places are the where array for
its what array (`this`: `t` at 0, `h` at 1, `i` at 2, `s` at 3, `th`
at 0, `hi` at 1, `is` at 2, `thi` at 0). The letters with their places
are the word's own characters, so the two arrays put a form back
exactly. *In the source today* the store keeps each form's characters
(`inverse_table`). *To note:* this is the arrangement of a thing's
parts, the second where that §19.48, answer 4, set aside for the
read; it returns here for the way back only.

### 19.58 Alec: the way back for generation too; the whole stores its parts in order (2026-11-19)

Alec, of point 4 of §19.57:

> The problem is that we are not always doing reconstruction: sometimes
> we do generation.
>
> As an alternative to location matching on words: the whole stores its
> parts in order, and longest first. That will give us reassembly
> instructions, but I think we want to keep the part matrices binary.
>
> What might also work is a network from words to parts with learnable
> reconstruction order. Lets do whatever is most efficient for
> reconstruction.

**His point.** §19.57 took the places of a form's parts from the read
that matched them. A word that is generated was not read, so there are
no such places. The way back has to start from the form alone.

**In the source today.** A form's row goes back to its characters by a
stored table (`RadixLayer.inverse_table`, read by `RadixLayer.reverse`;
its comment: "Exact, no learning"). It is a list on the host.

**Run by Claude** (197,014 forms; 20 words at a time; scratch script
`way_back_forms_check.py`). Three ways from a form to its characters,
none of which uses a place from a read:

| | Entries for a form | Exact | 20 words on Metal | on the host |
| --- | ---: | --- | ---: | ---: |
| 1. Its characters in order, a table | 8.7 | all 197,014 | 0.149 ms | 0.002 ms |
| 2. The same as binary rows, a block of 27 for each position | 8.7 | all 197,014 | 0.149 ms | 0.042 ms |
| 3. His alternative: its parts in order, the longest at each place, each then expanded to its characters | 4.3, at most 8 | all 197,014 | 0.162 ms | 0.010 ms |

- *His alternative is exact and needs nothing else.* The parts in
  order, the longest at each place, put back every form: no location
  arrays, no learned order.
- *In time there is nothing to choose on Metal;* all three are one or
  two gathers. In size his alternative is half of the others.
- *Binary.* The ordered list is binary as one block of the parts for
  each slot of the order, with a single one in each block. The read's
  matrix of a form's parts stays as it is, binary and with no order.
  They are two arrays: the read's holds every part inside the form
  (15.9), and the way back's holds only the parts that tile it (4.3).
- *Way 1 is his alternative with letters as the parts,* and is what the
  source already keeps.
- *The network with a learnable order* was not run. It is not needed to
  put back a form the store holds, and it can be wrong where a table
  cannot. It is what would be needed for a form that was never stored.

**Taken, on his word to do what is most efficient** (*Claude's choice;
his to overrule*). The whole stores its parts in order, the longest at
each place (way 3). It serves reconstruction and generation alike,
since it starts from the form. Point 4 of §19.57, the location arrays
taken from the read, is withdrawn.

### 19.59 Where item 1.7 stands, and the text to begin it in a new conversation (Claude, 2026-11-19)

Alec, 2026-11-19, of §19.52 to §19.58: "So this gives us a clean
perceptual basis spec for 1.7." And: "provide text for 1.7 to begin in
a new conversation with you. Please spin off implementers in Opus as
usual." The discussion is finished; the item's text can be written.

**What is decided, and where.**

| Matter | Decided | Section |
| --- | --- | --- |
| The layers: Pi, Not, Sigma at every order; symbols split from concepts | rules 1 to 32 | §19.21 |
| The looks: the bank, the request, the returned set, `attend`, dimming, one gain field, one where, a literal | rules 33 to 42 | §19.51 |
| The salience map is a gradient taken from the open run; "no derivative" of §19.52 is withdrawn | rule 36, open as to its form | §19.60 |
| The salience map selects the level, the `.where` selects the word; what stays selected is drawn and judged by its outcome | rule 37 restated | §19.53 |
| The request names a place in the conceptual tree; "words" gives the space a Not entry | rule 35 | §19.54, §19.57 |
| A form's code: no end marks; a space character; its parts of every size | Alec | §19.54 |
| The part map: from the letters up, a part of any length only where two forms would be the same; a part added is carried by every form that holds it | Alec; 762 parts for 197,014 forms by a greedy estimate | §19.56 |
| The parts of a sentence by one product for each character length, on the device | Alec; 0.50 ms a sentence on Metal | §19.55, §19.57 |
| The way back, for generation too: the whole stores its parts in order, the longest at each place | Alec's alternative, taken | §19.58 |
| Test points: the structure as composed is valid; a word among garbage letters; "words" lowers a form that holds a space | certificates | §19.53, §19.57 |

**Open, to settle in the item or with Alec.**

*Answered by Alec on 2026-11-19: §19.64. The rules in one place are
§21, and the item's text for the implementers is §22.*

0. The form of the gradient salience map (§19.60): the plain
   gradient misses words that are fully present; recommended, each
   Sigma passed before its squash.
1. The where as a pathway of its own (§19.62, §19.63: fixed
   wires, intersected with the what, no location concept made).
   Each percept has its own band of the code's wires.
1. How the word is found from its parts on the device without a
   product against every form (§19.57: 5.9 ms against 197,014 forms,
   growing with the dictionary).
2. A pair that is no known part leaves no trace (`felixx` read as
   `felix`), and a wrongly rebuilt word sometimes reads as the word
   meant (§19.55, §19.57): whether the length or a match at every
   place is added.
3. A read that gives no WFF (Alec: "we will probably have to return to
   this"). In item 1.8 it outputs nothing (1.8 §68.89). Alec,
   2026-11-19: with no WFF, NULLs are reconstructed, by a rule that
   applies automatically (1.8 §68.91; the rule `null` exists and
   nothing applies it). The mining of
   silent outputs with bisection is item 1.6's.
4. Rule 32, spellings: what a near spelling should read, with the
   width of a row's cone.
5. `PiLayer2` passes no gradient at a membership of exactly one
   (§19.50); and, not checked, whether the read to a point, clipped to
   0 to 1, can reach a root idea with a negative coordinate (1.8
   §68.87).

**The text to begin with** (for a new conversation with Claude; Alec
pastes it):

~~~
Begin item 1.7 of BasicModel (the Sigma and Pi layers, the looks, and
the perceptual basis). Read first, in
basicmodel/doc/plans/2026-11-14-sigma-pi-layers.md: section 19.59 (what
is decided and what is open, with the sections that hold each),
then 19.21 (rules 1 to 32) and 19.51 to 19.63. Read the item 1.7 entry
of basicmodel/todo.md. Your memory holds how we work.

Do, in this order:
1. Write the rules of item 1.7 once, as one clean statement of the
   perceptual basis and the looks (rules 1 to 42 as they now stand,
   with what 19.52 to 19.63 changed folded in), as a new section of
   that file. Mark what is mine, what is yours, and what is open.
2. Put the open points of 19.59 to me, each with what you would take.
   Do not hold the text for them: build on what you recommend unless I
   say otherwise.
3. Write the item's text for the implementers as one hand-off in that
   file (it replaces 19.8), in rounds, each with its targeted tests and
   its certificates, the test points of 19.53 and 19.57 among them.
   One default sweep at the item's push round; no slow tier.
4. Start the implementers in Opus, in the same checkout, and review
   each return against the source before the next round goes.

Item 1.8 is complete and on GitHub. Commit your own work locally; do
not push unless I say.
~~~

### 19.60 Alec: back to a gradient salience map, taken from the open run (2026-11-19)

Alec:

> By the way, I think we will have to revert to a gradient saliency map
> from the output, since it is input sensitive: IOW, we will need to
> use the parallel gradient from the open attention run to determine
> the operating point of the conceptual attention over the current
> input, which is something that the weights will not be sensitive to.
> So the right fix is probably not to use weights.

**What it changes.** Rule 36 as §19.52 restated it ("no derivative is
taken") is withdrawn. The salience map is a gradient of the asked
concepts by the input, and it is taken from the open run, the first
look, in which the whole sentence is read at once. *Claude's reading:*
the local derivatives are kept from that one pass, as §19.46's rule 36
had it, so the map costs no second backward pass and is in step with
the current input. His "probably" leaves the form of the fix open.

**What has been measured that bears on it** (Claude; the 304 requests
of §19.52, with rule 2's rows; `way_back_check.py`, with an integrated
gradient added today):

| The request carried down by | Returns exactly the asked words that are present | Misses a word that is fully present |
| --- | ---: | ---: |
| the gradient as it is (the Sigma by its slope) | 230 of 304 | 74 |
| an integrated gradient, from an empty field to the field read, 32 steps | 254 of 304 | 50 |
| the gradient with each Sigma taken before its squash (its sum, not its `tanh`) | 304 of 304 | 0 |
| the rows' entries alone, times what is active (no gradient) | 304 of 304 | 0 |

- *The fault of the plain gradient stands:* where the open run has lit
  a concept fully, its slope is near nothing (0.030 against a weight of
  3.0, §19.52), so the words that are most present are the ones
  missed. Whatever is taken has to deal with that.
- *The third row is a gradient, and it is sensitive to the input.* It
  passes each Pi by the derivative of its product at the values of the
  open run, and multiplies by the activity. What it leaves out is only
  the last squash of each Sigma. §19.51 called it "by its weights",
  which named it badly.
- *The fourth row is what he sets aside:* it reads the weights and what
  is active, and nothing of how strongly the current input drives each
  row.
- *Not measured:* cases where the two right-hand rows differ. On these
  sentences every part is fully present, so they agree. They would
  differ where parts are present by degree, which is his point.

**Rule 36 as it now stands** (*open; Claude's recommendation is the
third row*). The salience map is the gradient of the asked concepts by
the percepts of the sentence, from the open run's kept derivatives;
each Sigma is passed before its squash, so that a concept already lit
still returns its percepts. To be confirmed by him, and tried on
inputs where parts are present by degree, in the item's first round.

### 19.61 Alec: weights for the what, perhaps, but not for the where; how the where moves (2026-11-19)

Alec: "Maybe by the weights is OK to attend to percepts/concepts, but
not for locations? How is attention scheduled to specify the .where
that moves from one word to the next, and how are we updating that
based on current position in the sentence?"

**How the where moves, as things stand.**

- *Built, item 1.8.* The where is a bracket, an interval of the
  sentence. The first look is the whole sentence; each later look is
  one act of the chooser on the bracket table (`narrow_step`: `focus`,
  `unfocus`, the field's and, or, not). The chooser's context is the
  understood prefix: the occupied STM slots with their places and
  times, the heat of the three spaces, and each bracket's excess and
  omission (`AttentionContext.context`). A word that was interpreted
  marks its bracket done. The order is taught by the reading lesson
  (`AttentionLesson.ReadingTargets`). So the current position is held
  as the table's done marks and the places of what is in STM.
- *Tried, item 1.7's prototype* (§20.8, `order_trial`). The request of
  a look is the meta concept "word" and one place. The first place is
  the sentence's start; each later place is the end of the word read
  last plus a step. The place is given as its code on the sinusoid,
  made continuous, and a place row reads each percept at its own
  place. Only the step was taught, by the gradient of a loss that asks
  for the next word, through that code. It went from -3 to +1.07 in 20
  steps and then read all sixteen sentences in order.
- *In the rules* (§19.51, rule 38). No order is built in; reading in
  order is taught and a test of it gates the grammar tests. That the
  place comes from where the last word ended is Claude's note from the
  trial, not a rule.

**So his "not for locations" is how it already is, in the trial.** The
where of a look is not read from weights. It is a value `attend` gives
at each call, computed from the current position, and what is learned
is how to compute it. The what part is where the choice of §19.60
lies.

**Not designed.**

1. What `attend` reads to know the current position in item 1.7: the
   end of the word interpreted last (the trial), the places of what is
   in STM (1.8's context), or both.
2. The end of a word. A place's code holds its start only; the end is
   the start and the length of the form that matched.
3. What happens after a look that interprets nothing (the place must
   still move), and whether the done mark stays.
4. Places that are not the next word: a look back, or a wider look
   (`unfocus`).

### 19.62 Alec: a parallel pathway for the where, and location concepts (2026-11-19)

Alec, of §19.61:

> The 1.7 prototype of ""a word" plus one place" seems unbiological,
> lets do something more continuous if possible. I had mentioned a
> parallel pathway on .where, so that the .where we are currently
> stamping onto concepts gets muxed in from a parameter surface that
> can hold the back-projection. So the saliency is selecting both
> content and location by concepts that may have location information;
> that means attend will general select the word level, but will
> change its location concepts to search the next available based on
> the STM state.

**Decided** (Alec). A look is not "a word, and one place given as a
number". The where runs on a pathway of its own beside the what. The
`.where` that is stamped onto concepts today is instead muxed in from
a surface that can also hold the back-projection. The salience map
selects content and location together, by concepts, some of which
carry location. `attend` keeps asking for the word level in general
and changes the location concepts it asks for, so as to search for
the next available word, from the state of STM.

**Claude's reading, for him to correct.**

- *The surface* is an array over the places of the sentence, held
  beside the what. The way up reads the where from it; the way back
  (the salience map of §19.60) writes onto it. So where attention is
  directed is a field over places that changes by degree from one look
  to the next, and not one place.
- *Location concepts* are concepts whose rows have weight on the where
  pathway, as rule 39's place row has, so that asking for them is
  asking for a region.
- *The schedule* is then nothing of its own: the request is what
  `attend` gives from STM, and it moves because STM changed when a
  word was taken. This answers point 1 of §19.61: the current position
  is read from STM.
- *What it replaces:* the trial of §20.8 (the end of the last word
  plus a taught step) is a discrete stand-in and is not built.

**In the source today.** A subspace holds what, where and when either
as one muxed event or apart (`SubSpace.set_demuxed`, `mux`, `demux`),
and the where is written by position from the constant sinusoid
(`WhereEncoding.stamp`). So the two pathways exist as storage. What
does not exist is a where that anything but the position writes, and a
concept that asks for a region.

**Not designed.** What the surface holds at each place (the sinusoid
code, a weight on it, or both); whether it is a parameter that is
learned or a state of the sentence that is written at each look; how
the back-projection onto it is combined with the constant code; and
what a location concept's row is and how it is taught.

### 19.63 Alec: the where surface is fixed wires; the intersection; no location concept of its own (2026-11-19)

Alec, on the four points §19.62 left not designed:

> What the surface holds at each place: I think we just want to hold
> wires that correspond to the .where, so the saliency on that will
> tell us where to look. Is that biologically coincident?
>
> Whether it is a learned parameter or a state: It would be fixed, as
> are the positions getting stamped into the percepts on .where.
>
> How the back-projection is combined with the constant code:
> Hopefully this is clear; we are taking the intersection of what and
> where.
>
> What a location concept's row is and how it is taught: we dont need a
> specific location concept. concepts have the location to abstract
> over, and a gradient incentivizing them to do so, so it will learn
> location-concepts over the network, and attend() will expose that
> network gradient to the error.

**Decided** (Alec).

1. *The surface holds wires for the `.where`,* nothing else. The
   salience on those wires says where to look.
2. *It is fixed,* as the positions stamped into the percepts are
   (`WhereEncoding.stamp`). Nothing of it is learned and nothing is
   written into it as a state.
3. *The combination is the intersection of what and where*
   (§19.48, answer 2): a percept is in the look to the degree that
   both its what and its where are salient.
4. *No location concept is made.* Concepts already have the location
   among what they abstract over. The gradient gives them the reason
   to use it, so concepts of location form in the network, and
   `attend` exposes that gradient to the error.

**What goes.** Rule 39's place row, as a row built for the purpose
(§19.51, §20.5), and §19.62's "location concepts" as a kind of their
own. The four points §19.62 listed as not designed are settled.

**His question: is it biologically coincident?** In kind, yes, by two
well-known accounts; Claude read both pages on 2026-11-19.

- The what and the where run on two pathways that are distinct and
  linked: object identity on one, location on the other
  (https://en.wikipedia.org/wiki/Two-streams_hypothesis; the page also
  records the criticism that their independence is overstated).
- Features are held on maps tied to one master map of locations, and
  attending to a location gathers the features found there
  (https://en.wikipedia.org/wiki/Feature_integration_theory). Wires
  for the where, with salience over them selecting what is bound, is
  that arrangement.

*A wire for each place, or the wires of the code: no difference*
(Alec, 2026-11-19: "If i understand correctly, there is no
difference."). He is right, and Claude's distinction was mistaken. The
where is stamped onto each percept, so each percept has its own band of
the code's wires. The surface is that: one band for each percept. The
back-projection lands on a percept's own band, so salience on it is
salience at that percept's place, and the map is laid out place by
place as in the accounts above.

### 19.64 Alec's answers on the open points of §19.59 (2026-11-19), and what each changes

Alec answered the open points when the item was begun. His words are
quoted whole. He gave them under the numbers 1 to 6 of a list put to
him in conversation; §19.59's own list numbers the same points 0 to 5,
with two of them numbered 1, so each is named here by its subject.
What follows a quotation is Claude's reading unless it is marked
*decided*. What is marked *mine* is built as written and is his to
change.

**1. The form of the gradient salience map** (rule 36).

> You said weights were better, lets do that; I had not specced the
> what/where pathways at that point

*Decided.* The salience map passes each Sigma by its weights, which is
to pass it before its squash: the third row of §19.60's table, which
returned exactly the asked words that are present in 304 of 304
requests. It is still a gradient and it is still taken from the open
run: each Pi is passed by the derivative of its product at the values
of the open run, and the result is multiplied by the activity. His
"the right fix is probably not to use weights" (§19.60) was said
before the what and the where had their two pathways. The where now
has wires of its own for the map to light (point 7 below). *Still to
be tried,* as §19.60 said: inputs where parts are present by degree.
It is a step of the round that builds the looks.

**2. Finding the word from its parts without a product against every
form.**

> we might as well move to the powerset version we developed if that
> will resolve this. Up to you when we do that.

*Claude's reading of "the powerset version":* the part map of §19.54
to §19.57, in which a form's code is the set of the parts inside it,
one point of the powerset over the existing parts.

*Does it resolve it? Yes, when the set is used as the form's address.*
The 5.9 ms of §19.57 was measured with the part map: it is what it
costs to multiply the parts found against the parts of every form.
Since no two forms have the same set of parts, the set itself says
which form it is, and nothing has to be multiplied. Run by Claude
today (the 197,014 forms and 762 parts of §19.57, sentences of 20
words; scratch script `set_address_check.py`). Each part has two fixed
whole numbers. The numbers of the parts found inside a run of letters
are summed, which is one small product, and the two sums are one key.
The key is looked up in a sorted table of the forms' keys, and the
form's own parts are then compared, so a match is exact.

| The word, from the parts found | On Metal | On the host | Stored |
| --- | ---: | ---: | ---: |
| the parts multiplied against every form (§19.57) | 5.89 ms | 8.43 ms | 600 MB as a dense matrix |
| the set of parts as an address, one probe | 0.45 ms | 0.34 ms | 49 MB |
| both stages: the parts at every place, then the word by one probe | 0.76 ms | 0.72 ms | |

- The two ways give the same answer at 500 of 500 runs of letters, the
  right word each time.
- They read the strings that are no form alike: `felixx` as `felix`
  and `pimpleprooof` as `pimpleproof` (point 3 below); `caat`, `zzz`,
  `xqzcatpl` and `thisis` as no word.
- Two of the 197,014 forms have the same key as another form. The
  comparison of the parts tells them apart.
- The probe does not grow with the dictionary as the product does: a
  sorted table of twice the size costs one step more.

*When* (*mine*, on his "Up to you"). Two things are meant by "moving".

- *The code of a form* is this item's. It is what gives exact parthood,
  and rules 5 and 43 of §21 build it.
- *The lookup that finds a sentence's forms* moves to the device in
  item 1.5, as its plan has it (results equal to the path before it;
  its section "The radix trie as a tensor"), and it moves there in
  this form: the parts at every place by one product for each
  character length, then the word by its set of parts as an address.
  It is rule 44 of §21.
- So in this item the store's one lookup at the start of a sentence
  stays where it is, and the first Pi is read only on the rows it
  names (rule 33). Nothing built in this item multiplies against every
  form. Recorded in the item 1.5 plan and in `todo.md`.

**3. A pair that is no known part leaves no trace** (`felixx` read as
`felix`).

> We can teach that difference when we observe it in the output. so
> just an idempotent teaching round, gathered from the training
> harness logs

*Decided.* Nothing is added to a form's code or to the match for this:
no length that has to agree, and no match at every place. Where a
string is read as a word it is not, the output shows it, and the
difference is taught then. The teaching is a round over cases gathered
from the training harness's logs. It is idempotent: run a second time
on the same cases, it changes nothing.

*Claude's reading of the teaching.* The string that was read as
another word is admitted as a form of its own. By the part map it then
differs from that word in at least one part (`felixx` and `felix`
differ by `xx`), and its own row, the larger of the two, is the one
taken at its place (rule 37). Admitting a form that is already known
changes nothing, and that is what makes the round idempotent.

*In the source today.* The harness does not log it.
`bin/corpus_lessons.py` writes one row for each epoch (`epochs.jsonl`).
The input is compared with what is put back once, at the end of a
trial, on a row's whole list of words (`Models._reconstructionReport`).

*Where it is built* (*mine*). The store's side is this item's and has
a certificate (rule 46 of §21). The log of a word put back as another
word, and the round that gathers those cases, go with item 1.6's
harness, beside the mining of silent outputs that 1.8 §68.89 placed
there. Recorded in `todo.md` under item 1.6.

**4. A read that gives no WFF.**

> When the parse fails, we can just write an empty null-terminated
> sentence by force by calling silence() : S → NULL (which is not
> actually in the grammar becuase it is only a fallback for a broken
> grammar) and then terminating.

*Decided.* When the read of a sentence gives no well-formed formula,
the sentence is written by force as an empty sentence that ends in
NULL, by a rule `silence`, S → NULL, and the read ends there. The rule
is in no grammar file: it is a fallback for a grammar that is broken,
and nothing chooses it. It is the rule "that applies automatically
when we fail to find a WFF" of 1.8 §68.91, now named.

*In the source today.*

- A sentence whose trees do not reduce to one is left with no row.
  `Models._stm_reduce_to_single_S` tags the depth that is left over as
  negative ("so it cannot become a program or a memory row"),
  `_program_entries` gives `None` for it, and `_commit_sentence` skips
  the write. Nothing is said, and the standalone kernel's head gives
  0.5 in every place (1.8 §68.90).
- A rule `null` exists (`NullLayer`, `bin/Language.py`), declared in
  `data/ladder.grammar` alone. It is the identity on one operand, and
  it is not this rule.
- `silence` names no rule today. The word is used for an answer that
  says nothing (`What.silence`).
- A rule's layer is found by its name
  (`SymbolSubSpace._resolve_rule_layer`, from `GRAMMAR_LAYER_CLASSES`),
  so a layer can be registered that no grammar file declares.

It is rule 48 of §21, with Claude's readings marked there.

**5. Spellings** (rule 32).

> I don't think we need to specify those, they just do what the
> architecture currently dictates. lets see.

*Decided.* Nothing more is specified for rule 32: not what a near
spelling reads, not which quantity is the width of a row's cone, and
not how a hit or a miss moves it. A row of the first Pi does what
rules 2 and 15 give it, and what a near spelling reads is reported
(certificate 27 (c) of §21). Points 8 and 10 of §19.21's open list
close with this.

**6. Two points of the source.**

> I am open to your suggestions on both

*`PiLayer2` passes no gradient at a membership of exactly one.* Its
clamp is `m.clamp(EPS_LOG, 1.0)` (`PiLayer2._clamp_membership`); the
ceiling is there so that the logarithm never rises above zero.

- *Checked today:* nothing in `bin/` builds a `PiLayer2` or a
  `SigmaLayer2` but `bin/Layers.py` itself: `SigmaLayer2`'s own
  kernel, and `MeronymicFoldAdapter`, which nothing in `bin/` builds
  (tests do). So no training of today passes through that clamp.
  §19.52's "training passes through it" was wrong.
- *Suggested, and built so:* `PiLayer2` is left as it is (rule 1).
  `PiLayer3` gives the same values and passes the derivative wherever
  the membership lies between the floor and one, one included; below
  the floor it passes none, as now. Certificate 2 tests both.

*The read to a point, clipped to 0 to 1.* `SigmaPiStack.forward` clips
its input and each Sigma's result to that range (`_field_value`), and
the read's teacher is the kept reading's root idea
(`IdeaRead.complete`, `record.meaning_root`).

- *Checked today, on a small case only:* in the two reads of
  `test/test_item18_idea_read.py` the root idea's coordinates lie
  between 0 and 0.4, and none is below zero. *Read, not traced:* the
  grammar's maps are charted on the unit cube (`cube_chart`, which
  asserts the range). So the clip costs nothing as things are.
- *Suggested, and built so:* under rule 18 the stacks end in rule 3's
  Sigma, one signed number in -1 to 1, and the clip goes with the old
  arithmetic. Certificate 7 prints the least and the greatest
  coordinate of the root ideas of the lesson's sixteen sentences
  beside the range the read can give, and fails if a teacher's
  coordinate lies outside it.

**7. The where as a pathway of its own** (§19.62, §19.63).

> I guess I did not address the where as a pathway of its own. But
> since we are currently "stamping" the .where, if we just let that
> flow from a pytorch param with the sinusoid written to it, the
> saliency map should light it up.

*Decided.* The `.where` that is stamped onto percepts flows from a
parameter that holds the sinusoid. The salience map is a gradient, so
it lands on that parameter as it lands on any wire, and what it lights
there is where to look.

*In the source today.* The code of a place is computed when it is
asked for, from the place's whole number (`WhereEncoding.encode`, read
through `WhereRegistry.encode`). It is no parameter and no buffer, and
no gradient reaches it. It is written onto the sentence's percepts in
`PartSpace._radix_code_events` and `PartSpace._embed_radix`.

It is rule 47 of §21. *Mine there:* the parameter is in no optimizer's
groups, since §19.63 has it fixed; a percept's band is its row of the
parameter, taken by the percept's place; and the arithmetic of the
where match, which neither §19.63 nor this answer gives.

**What is open after these answers** is listed once, at the end of
§21, each point with what is built meanwhile.

### 19.65 Alec on four of Claude's choices in §21 (2026-11-19)

Claude put to him the choices of §21 that he might overrule. He
answered four, by the numbers of that list, and added a paragraph on
how location is taught.

**The teaching of a string read as a word it is not** (rule 46).

> We don't want to admit Felixx, we want to admit the part that
> discriminates it from Felix: "xx".

*Decided.* The teaching admits a part, not the string. Claude's reading
in §19.64, point 3, that the string is admitted as a form of its own,
is withdrawn. *Mine:* the part is the shortest run that the string
holds and the word does not, which is rule 43's own step. The string
gets no row. Once the part exists, the parts found in `felixx` are no
longer exactly the parts of `felix`, so it is not read as `felix`
wherever a word is the form whose parts are exactly those found
(§19.55; rule 44). Admitting a part that exists changes nothing, so
the round is idempotent as before.

**The where parameter** (rule 47; Claude had it in no optimizer's
groups).

> If we are reading over weights, they need to be optimized, though.

*Two readings, and Claude has asked which.* (a) The parameter's own
values are changed by training. That would take back §19.63, point 2
("It would be fixed, as are the positions getting stamped into the
percepts"). (b) The weights that the salience map is read over have to
be trained weights, which the concepts' weights on the where are. *A
fact that bears on it:* a gradient reaches a tensor that takes one
whether or not an optimizer holds it, so the salience map lights the
parameter either way; an optimizer is needed only to change its
values. *Meanwhile:* round 2 tries it both ways, and round 6 builds
what he says.

**"Objects", beside "words"** (rule 35).

> Since we create the meta by hand, we might as well add the "objects"
> concept also.

*Decided.* A concept "objects" is declared as "words" is: one concept
that every object's concept is a member of. *Mine:* an object here is
what `interpret` finds or makes for a word, the blank object among
them (rule 20).

**How location is taught.**

> Reading in order will create location concepts as long as we are
> teaching location. So we might teach the reconstruction over bytes
> as built through a conjunction of the what and where pathways;
> recreated percepts have recreated locations that place them in the
> out buffer, so the error is over both pathways.

*Decided, as a direction ("we might").* Location concepts come from
teaching location, and the reconstruction is a teacher of it. The
bytes that are put back are built through a conjunction of the what
pathway and the where pathway: a percept that is recreated has a
location that is recreated with it, and that location is what places
it in the out buffer. So the reconstruction's error is over both
pathways. It is rule 49 of §21. *Mine there:* the form of the
conjunction, and where the source places what it puts back today.
This answers the second half of §21's open point 1 in kind: whether
location is learned depends on its being taught, and this is the
teacher. Round 2 tries it beside the reading lesson.

### 19.66 Alec: the where parameter stays fixed; one where tensor over subjective space (2026-11-19)

Asked which of §19.65's two readings he meant, Alec:

> The parameter itself must stay fixed if the sinusoid coding is to
> remain meaningful across all .where
> And probably this indicates we want a single .where tensor that
> covers subjective space. If we let it vary, then we can have a
> nonlinear representation over the somatosensory cortex, which is
> what gives the somatosensory homunculus big hands and face. Correct?

**Decided.** The parameter's values are not changed by training.
§19.63, point 2, stands, and the open point of §19.65 is closed. What
is trained is what reads the where: the concepts' weights on it.

**One where tensor over subjective space** (his "probably"). *In the
source it is nearly so already:* one registry gives every space its
places from one address range (`WhereRegistry`: disjoint slices for
text, numbers, scalars, parts, wholes, symbols and concepts) and one
encoder codes them all (`install_where_registry` puts the same
`WhereEncoding` on every subspace). Rule 47's parameter is then that
one tensor: one row for each address of the registry. *Mine:* built so
in round 6.

**His question, and Claude's answer: yes, in kind.** In the body's map
on the cortex the hands and the face take far more surface than their
skin's area, in proportion to how finely they tell touches apart. The
same holds in vision, where the centre of the eye takes most of the
first visual area. It is called cortical magnification (from memory;
the pages were not read today). A where code that is allowed to vary
could come to spend more of itself where finer telling apart is
needed, which is that. *Two things to keep apart:* (1) the code of a
place, which has to mean the same wherever it is read, and so stays
fixed, as he says; (2) how much of the code's range a region of
subjective space is given. The second can vary with the first fixed:
a place is passed through a learned map that keeps order before the
fixed sinusoid codes it, so a region that matters is spread over more
of the range. Nothing of it is built in this item.

*Alec, the same day:* "That future work should indicate that the map
is laterally connected as a SOM, so that it remains continuous." A
self-organizing map ties each unit to its neighbours, so that a unit
that learns pulls its neighbours with it and places that are next to
one another stay so. It is in `doc/FutureWork.md` ("A where that gives
more of its range to what matters"), with what is left to specify.

## 20. The prototype of the looks, in tests only (Claude, standing in for Codex, 2026-11-18)

This returns the text of §19.47, as amended by §19.48 and §19.49. No file of `bin/` is changed and no existing test; nothing is wired into the model. It decides nothing.

### 20.1 The files (Claude, standing in for Codex, 2026-11-18)

- `test/looks_prototype.py`. The bank of a sentence (rule 33): each byte's row in a what codebook of the lesson's bytes, its place and that place's code from `Spaces.WhereEncoding` (the constant sinusoid, both rungs, capacity 4096), and the stem's token it belongs to (runs of letters, each mark on its own). The first layer (rule 2 of §19.21): rows are products over a token's memberships, `z = exp(W log clamp(m, ε, 1))`, weights at zero or more. A word row has weight on its letters' codebook rows (the count of each); the fixed word whole has weight one on every byte of the token being a letter; a place row has weight on both unsigned lanes of the token's place code, `k·c` on the code `c` and `k·(1 − c)` on its complement, so it peaks at the place and its radius narrows as `k` grows. The Sigma (rule 3): each concept is `tanh` of a weight times its row; the meta concept "word" is one row over the word concepts. The pullback (rule 36): the share of a byte is its activity times the derivative of the asked concepts by it, by the transposes of the same matrices with the local derivatives kept from the way up, and by the framework's backward. The two domains (§19.48, answer 2): the asked word, whole and meta rows give each byte its what match (its share, capped at one); the asked place rows, read at each byte's place, give its where match; the look's activity is their product. The returned set (rule 37) is the way up again on that activity; `interpret` takes the most active word concept, exactly. Also the fallback, the sinusoid profile and the trial of reading in order.
- `test/looks_prototype_report.py`, which prints every table below.
- `test/test_item17_looks_prototype.py`, nine tests of the claims below (9 passed, 3.9 s).

Three choices that the rules do not make were needed to run it; each is reported against its alternative below. (1) The Sigma is passed in the pullback by its slope (rule 36 as written) or by its weights alone. (2) A token's membership of a codebook row is the greatest of its copies, and in the pullback every copy at that greatest value takes the whole derivative (the framework's `amax` splits it among the copies). (3) The share becomes the look's activity capped at full presence, `min(1, share)`.

### 20.2 The pullback by the transposes is the framework's backward (Claude, standing in for Codex, 2026-11-18)

On the sixteen sentences, at full activity and at three random activities, for four requests:

| Sigma passed by | Comparisons | Largest difference |
| --- | ---: | ---: |
| its slope | 256 | 2.8e-17 |
| its weights | 256 | 0.0 |

They agree once both use the same local derivatives. Out of the box they did not, for two reasons. `PiLayer2` (as `clamp(m, ε, 1)`) passes **no gradient at a membership of exactly one** in this torch (2.15): `PiLayer2(3, 3)` at `m = 1` gives a gradient of 0, and at `m = 0.999` about 1.01. A word wholly present stands exactly there, so the framework's backward through the Pi returns no share for it. The prototype clamps only the floor (an activity here never exceeds one). And the framework's `amax` splits a repeated letter's derivative between its copies, so `that` came back at a quarter.

### 20.3 Five requests on the lesson's sixteen sentences (Claude, standing in for Codex, 2026-11-18)

The word meant: for a word, the sentence's last word; for the place, the word at token 1; for the meta concept with a place, the word at token 2. Place rows at concentration 200, applied inside the products. "Projection" is the weights alone: each byte whose letter is in an asked row.

| Request | Sigma passed by | Place read at | Taken as meant, first of equals | Taken as meant, largest of equals | Taken with a tie | Returned on "This is Felix." (token: word concepts at 0.5 or more) | Projection on "This is Felix." |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| the meta concept | slope | — | — | — | 0 of 16 | nothing | 0:This 1:is 2:Felix |
| the meta concept | weights | — | — | — | 16 of 16 | 0:This/is 1:is 2:Felix | 0:This 1:is 2:Felix |
| a word's concept | weights | — | 10 of 16 | 14 of 16 | 6 of 16 | 2:Felix | 0:This 1:is 2:Felix |
| the fixed word whole | weights | — | — | — | 16 of 16 | 0:This/is 1:is 2:Felix | 0:This 1:is 2:Felix |
| an absent word (`dog`) | weights | — | 16 of 16 | 16 of 16 | 0 of 16 | nothing | nothing (on "Who is here?", `Who` by its `o`) |
| a place | — | each byte | 16 of 16 | 16 of 16 | 0 of 16 | 1:is | all tokens |
| a place | — | the token | 16 of 16 | 16 of 16 | 0 of 16 | 1:is | all tokens |
| meta concept and a place | weights | each byte | 8 of 16 | 8 of 16 | 0 of 16 | 1:is (Felix meant) | 0:This 1:is 2:Felix |
| meta concept and a place | weights | the token | 12 of 16 | **16 of 16** | 4 of 16 | 2:Felix | 0:This 1:is 2:Felix |

What comes back is what was asked for and is present: an absent word returns nothing in all sixteen, where the weights alone let a word in on one shared letter. The meta concept returns every word, and with a place read at the token and the largest of equals taken, the word at the place in all sixteen, as §19.49 expects. The ties are the containment of §20.4.

### 20.4 Cost, and what the binding by token does and does not prevent (Claude, standing in for Codex, 2026-11-18)

| One way up | The pullback | One look (the pullback and the way up again) |
| ---: | ---: | ---: |
| 478 | 600 | 1,078 |

Multiply-adds per sentence, the mean over the sixteen, counted from the shapes (14 word rows, 20 codebook rows, 4 tokens). A look costs about 2.3 ways up, once for a request and not once for each candidate.

| Sentence | Request | Returned | Taken |
| --- | --- | --- | --- |
| "Fe li x is here." (every letter of Felix, in three tokens) | Felix | nothing | nothing |
| "xileF is here." (the same letters in another order) | Felix | 0:Felix | Felix at token 0 |
| "This is Felix." | is | 0:is 1:is | is at token 0, tied with is at token 1 |
| "What is that?" | that | 0:that 2:that | that at token 0, tied with that at token 2 (`What` holds t, h and a) |

The binding by token stops letters spread over tokens. It does not see order inside a token: an anagram is the word, and a word whose letters are all in a longer token (`is` in `This`, `here` in `there`, `that` in `What`) is wholly present there. A product row is a test of containment, not of equality.

### 20.5 Dimming by the asking row (Claude, standing in for Codex, 2026-11-18)

The meta concept with a place at token 2 (`Felix`, first byte 8) on "This is Felix.": the most active word concept at each token as the place row's concentration narrows it, with the where match read at each byte's own place or at its token's first place, and applied inside the products or once to a token's reading.

| Concentration | Read at | Applied | This (token 0) | is (token 1) | Felix (token 2) | Taken |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 12.5 | each byte | inside | 0.932 | 0.991 | 0.984 | is |
| 12.5 | the token | inside | 0.255 | 0.989 | 0.995 | Felix |
| 12.5 | each byte | once | 0.293 | 0.924 | 0.995 | Felix |
| 50 | each byte | inside | 0.285 | 0.962 | 0.853 | is |
| 50 | the token | inside | 0.000 | 0.931 | 0.995 | Felix |
| 200 | each byte | inside | 0.000 | 0.506 | 0.096 | is |
| 200 | the token | inside | 0.000 | 0.277 | 0.995 | Felix |
| 200 | each byte | once | 0.000 | 0.307 | 0.995 | Felix |
| 800 | each byte | inside | 0.000 | 0.004 | 0.000 | is |
| 800 | the token | inside | 0.000 | 0.000 | 0.995 | Felix |
| 3,200 | each byte | inside | 0.000 | 0.000 | 0.000 | nothing |
| 3,200 | the token | inside | 0.000 | 0.000 | 0.995 | Felix |

(The full table, with "once" at the token's place too, is printed by the report; "once" at either place takes Felix at every concentration.) Read at each byte's own place, the dimming cuts the word that was asked for: a place's code holds its start and not its end, so the word's later letters lie away from the place and are dimmed, inside the product, once each. A short word near the place then wins, and at 3,200 nothing is left. Read at the token's first place, or applied once to the token's reading, the word at the place is taken at every concentration, and the other tokens fall from about 0.9 to nothing as the cone narrows. The row's value at its own place, its magnitude, falls as `exp(−k·H)` (H the entropy of its code); the Pi may not raise it (its bias is at zero or below), so the prototype gives the Sigma's weight on a place row the inverse of the row's magnitude.

### 20.6 The where sinusoid as a coordinate (Claude, standing in for Codex, 2026-11-18)

A place row at byte 200, concentration 200: its reading at a distance, over its reading at the place.

| Capacity | Rung | +1 | +2 | +4 | +8 | +64 | +128 | +255 | +256 | +512 | +4096 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | long | 1.000 | 1.000 | 0.998 | 0.993 | 0.625 | 0.158 | 0.001 | 0.001 | 0.000 | 0.000 |
| 4096 | short | 0.888 | 0.627 | 0.162 | 0.001 | 0.000 | 0.000 | 0.884 | 1.000 | 1.000 | 1.000 |
| 4096 | both | 0.888 | 0.627 | 0.162 | 0.001 | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 |
| 64 | both | 0.788 | 0.389 | 0.024 | 0.000 | 0.000 | 0.000 | 0.783 | 1.000 | 1.000 | 1.000 |

At capacity 4096 the long rung's period is 8,192 bytes and the short rung's 256. Within a sentence the long rung separates nothing (0.998 four bytes away); the short rung does all of it, and alone it lets in every place a multiple of 256 bytes away, which the long rung then removes. At capacity 64 `WhereEncoding` gives both rungs the same period of 256, and nothing removes the repeats. At concentration 200 a place four bytes away reads 0.16 and one byte away 0.89; the words of a sentence begin three to six bytes apart.

### 20.7 The fallback (Claude, standing in for Codex, 2026-11-18)

A `.where` given outright; from that extent the largest word, by extent (§19.30 leaves "largest" open); failing a word, the largest concept, here the word whole.

| Sentence | Extent | Taken |
| --- | --- | --- |
| "This is Felix." | "This" | the word `This` |
| "This is Felix." | "This is" | the word `This` |
| "This is Felix." | "Felix." | the word `Felix` |
| "This is Felix." | "." | nothing |
| "This is Felix." | the whole sentence | the word `Felix` |
| "Who is here?" | "is here?" | the word `here` |
| "Fe li x is here." | "Fe li " | no word; the whole `li` |

It agrees with the returned sets of §20.3 wherever one word is at the place. It needs no pullback; it also needs no request, so nothing in it can be taught to ask.

### 20.8 Reading in order, taught on the request alone (Claude, standing in for Codex, 2026-11-18)

Each look asks for the meta concept and a place: the first at the sentence's start, each later one at the end of the word read last plus an offset. Only the offset is taught, by the gradient of a loss that asks the next word to be the most active, through the pullback, the place code (made continuous) and the way up. It starts at −3, inside the word just read.

| Steps | Offset | Sentences read wholly in order |
| ---: | ---: | ---: |
| 0 | −3.00 | 0 of 16 |
| 20 | 0.83 | 16 of 16 |
| 150 | 1.07 | 16 of 16 |

It learns to step one byte past the word read, onto the space before the next word, and then reads all sixteen in order. What was taught is one number shared by all sentences; nothing about which concepts to ask for was taught, and no done mark was used.

### 20.9 What surprised me (Claude, standing in for Codex, 2026-11-18)

1. *A saturated Sigma gives its strongest token almost nothing.* With the meta concept pulled back by the Sigma's slope, the token holding two whole words (`This`, and `is` inside it) gets about 1/300 of the share of a token holding one, and at full presence nothing is returned at all. The derivative is smallest where the concept is most lit. Passed by its weights, every letter of a present word has a share of at least one.
2. *The framework's backward through `PiLayer2` is zero at full membership* (§20.2). Rule 36's default, the transposes with the local derivatives kept, avoids it only if the derivative at the ceiling is defined to pass.
3. *A product row is containment.* Ties at the full value are the rule, not the exception: in every sentence for the meta concept and the whole, and in 6 of 16 for a single word. The largest of equals (§19.30) settles all but `that` inside `What`.
4. *The asked word is cut by its own place.* Dimming at each byte's place, inside the product, favours a short word beside the place over the word at it (§20.5).
5. *The long rung does nothing inside a sentence,* and at the small capacity the two rungs are one.

### 20.10 What I would change in rules 33 to 42 (Claude, standing in for Codex, 2026-11-18)

Each is *mine*, for Alec to take or not.

- **Rule 34.** Say how a share becomes the look's activity. *Default:* the share capped at full presence; a byte shared by two asked rows is not counted twice.
- **Rule 36.** The pullback passes the Pi by its product's derivative and the Sigma by its weights, not its slope; the greatest of a token's copies of a letter is a selection, every copy at it taking the whole derivative; at full membership the derivative passes. The transposes stay the default; the framework's backward is used only with those same local derivatives.
- **Rule 37, with §19.49.** `interpret` takes the most active word concept; among equals the largest (§19.30), then the one found first.
- **Rule 39.** The place row's dimming is read at a token's first place (or applied once to its reading), never at each byte's own place inside the product. The row's magnitude is carried by the Sigma's weight on it.
- **Rule 41.** One where, both rungs, and say the capacity: with the registry's capacity the short rung separates a sentence's words and the long rung removes the short rung's repeats. And the binding by token does not know order, so anagrams and letters contained in a longer word read as the word; either the ordered run that §19.48 answer 4 set aside returns, or a word row takes its letters at their places within the token.
- **Rules 35 and 38.** The trial of §20.8 suggests what `attend` learns first is small: where the next place is, from where the last word ended.

### 20.11 Test files run (Claude, standing in for Codex, 2026-11-18)

`test/test_item17_looks_prototype.py` (9 passed, 3.9 s). The tables are `test/looks_prototype_report.py`'s output (deterministic; its random activities use a generator of its own). Item 1.8's third default sweep (1.8 §66.15) included `test_item17_looks_prototype.py` and ended with no failure.

## 21. Item 1.7: the rules in one place (Claude, 2026-11-19)

Alec, 2026-11-19: "Write the rules of item 1.7 once, as one clean
statement of the perceptual basis and the looks ... Mark what is mine,
what is yours, and what is open." This section states every rule of
item 1.7 once, as it stands after §19.52 to §19.64. For the build it
replaces §19.21 and §19.51; those sections and the ones around them
are its record, and where they differ this section stands. Nothing
here is new: each rule names where it was decided.

**The marks.**

- *(Alec, §n)* after a rule's name: the rule is his decision, and the
  section holds his words.
- *Mine:* a reading or a default of Claude's. It is built as written
  and is his to change.
- *Open:* not settled. What is built meanwhile is said, and the round
  that builds it reports it.

Rules 1 to 42 keep the numbers they had. Rules 43 to 49 are new
numbers for what §19.53 to §19.65 decided.

**What it builds.** One structure at every order: a Pi, then Not, then
a Sigma. Percepts, and above order zero symbols, are unsigned rows of
a square, sparse Pi, a product of memberships. Concepts are signed and
are a Sigma over them, each entry for or Not. A row exists only when
MTM nominates it. Symbols are split from concepts and have no
dimensions. A form's code is the set of its parts, so that one form is
a part of another exactly when its code lies inside the other's. A
sentence's percepts are read once into one bank, and a look is an
activity over that bank: `attend` asks for concepts, the request is
carried down as a salience map onto what each percept is and onto
where it stands, and what the way up then gives is what `interpret`
chooses from. The old classes are not edited.

**A. The layers and the structure.**

1. *New classes* (Alec, §18.24, rule 32). `PiLayer3` and `SigmaLayer3`
   (*mine:* the names). `SigmaLayer`, `PiLayer`, `PiLayer2` and
   `SigmaLayer2` are not edited, and nothing built with them changes
   value.
2. *The Pi* (Alec, §18.24, rules 21 and 31). Square and sparse, on
   unsigned values: $z = \exp(W \log \mathrm{clamp}(m, \epsilon, 1) + b)$,
   with `PiLayer2`'s constraints (diagonal at one or more, the rest at
   zero or more, bias at zero or less), so a row requires the members
   it has weight on and never exceeds its own reading. Its way back is
   the exact LDU solve. *Mine:* each percept or symbol is two unsigned
   rows, one for each lane (a reading of §18.25); a row has weight
   only on rows of its own lane (a reading of rule 29); and its
   derivative passes wherever a membership lies between the floor and
   one, one included, where `PiLayer2`'s clamp passes none at exactly
   one (§19.64, point 6).
3. *The Sigma* (Alec, §18.24, rule 23; §18.25; rule 29; "Signed sigma;
   good, this is knowing in the signed hyperball"). For each concept
   one signed number, $\tanh(x\,U)$, in $[-1, 1]$. Each input has one
   signed weight. Its positive part is the input's *for entry* and its
   negative part is its *Not entry*, so an entry is never both. A
   weight starts at zero or above (rules 8b and 21), so there is no
   Not entry until teaching makes one. The concept's value is stored
   as two unsigned lanes, its positive part on the for lane and its
   negative part on the against lane; a single reading is never
   "both". *Mine:* it has no bias; where an answer is read from a
   Sigma, the signed number is read.
4. *The cycle* (Alec, §18.25, as amended by rules 20, 21 and 26). At
   an order with $n$ concepts: $n$ signed values, stored as $2n$
   unsigned lanes; the rows above that MTM has nominated read those
   lanes, each row its own lane; each symbol's pair is combined, for
   less against, into one signed input; the next Sigma reads those
   inputs.
29. *One structure: Pi, Not, Sigma, at every order* (Alec, §19.20).
    - *At the basic level* the Pi makes the basic percepts, their
      vectors are combined (rule 7), and the Sigma is dense over the
      content's elements (rule 8). A Not entry there is from one
      element to one concept, a scalar. It records that something
      expected is absent: "animal cognition sees particular events
      not happening, but it does not negate ideas". What teaches it
      is an omission, the concept expected and not come ("Omission
      teaches not: yes, for percepts, the opposite of 1 is 0, not -1,
      as in Boole").
    - *Above it,* for a mind that has symbols, the structure repeats:
      a Pi makes symbols over concepts, a Not entry is from one
      symbol to one concept (the concept reads that symbol's pair
      swapped), and a Sigma makes the concepts. What tells a human
      mind from an animal's is this recursion, which symbols enable,
      and not the Not.
    - The operator `not` of a sentence or a thought swaps the lanes
      of what it is put on (`FieldLogic.reduce`), as now. It can be
      put on any symbol, and no omission need teach it.
    - With every weight held at zero or above, so that no Not entry
      is read, a mind speaks only of what is (§19.15).

**B. The perceptual basis: a form, its code and its way back.**

5. *A percept's code obeys the mereology* (Alec, §18.24, rule 27): at
   or above each part's code and at or below each whole's. For a form
   it is so by construction, since a form's code is the join of its
   parts' codes (rule 43). *Mine:* where a row is not a form coded by
   its parts (a configuration with `identityByParts` off, or the
   fallback by bytes), its code is the greatest over its parts' codes
   and a small draw keyed to the row, in place of the sum of its
   bytes' codes (`BytesFallbackEncoder.encode`), as §19.21 had it.
43. *The part map: a form's code is the set of its parts* (Alec,
    §19.54, §19.56).
    - A part is a run of adjacent characters. The parts of a form are
      its characters and the longer parts that exist and lie inside
      it.
    - *The space is a character,* a part like a letter, "so that the
      saliency map can select against it from within parts". A form
      may hold one (`ice cream`).
    - *No end marks.* A form's code does not mark its first and its
      last character. So a form has one code wherever it stands: `is`
      alone and `is` inside `this`.
    - *A longer part exists only where it is needed* ("we should also
      do pairs only when needed"). The map starts from the
      characters. A part of any length, pairs included, is added only
      where two forms would otherwise have the same parts.
    - *A part once added is carried by every form that holds that
      run,* the forms admitted before it and those admitted after.
      Otherwise a form with the part would stop reading as inside a
      longer form that holds it.
    - So parthood is read from the codes: a form found inside a form
      lies wholly inside its code. What makes something a word is no
      longer in the code; it is the level a request asks for (rule
      35).
    - *Mine:* the part added for two forms is the shortest run that
      one of them holds and the other does not. The exact code is
      the list of the form's parts (`WordIdentity.words`), and the
      row of fixed width is its fold, each part's fixed bits joined
      (`WordIdentity.atom_code`, `form`), in the layout's width as it
      is: no configuration is resized. Where two forms' rows are
      still the same after a part is added, the part's bits are
      raised as `WordIdentity._mint` raises a mint's today.
    - *Mine, and open:* the lengths 1 to n of today's code go. A
      length is not a part, and with the lengths kept `banana` and
      `bananana` would need no part to tell them apart, where his
      example adds one. The 32 coordinates of the length block stay
      in the layout and hold nothing for a form. *Open:* what they
      become (§19.54, point 3).
6. *A basic percept is a row of the Pi* over its parts and wholes
   (Alec, §18.5, rule 8, with rule 31), in place of the least of
   `BasicPercepts.read`. For order zero, `store.conjunctive` and the
   key `conceptualPi` go. It becomes a row when MTM nominates it (rule
   20).
44. *The lookup of a sentence's forms, on the device* (Alec, §19.55,
    §19.57; §19.64, point 2). The parts of a sentence are found at
    every place by one product for each character length. The word is
    then found from its parts by its set of parts as an address: one
    probe, and no product against every form (0.76 ms a sentence for
    both stages on Metal, against 6.25 ms). *Mine, on his "Up to you
    when":* this is built in item 1.5, with results equal to this
    item's. In this item the store's lookup stays as it is
    (`RadixLayer.lookup_with_id`: the dictionary, the trie's longest
    match, the fallback by bytes), it is made once for a sentence,
    and the first Pi is read only on the rows it names (rule 33).
45. *The way back of a form starts from the form alone* (Alec,
    §19.58: "we are not always doing reconstruction: sometimes we do
    generation"). No place of a part is taken from a read. "The whole
    stores its parts in order, and longest first"; the read's matrix
    of a form's parts stays binary and has no order. *Mine:* in this
    item the stored order is the table the source keeps, a form's
    characters in order (`RadixLayer.inverse_table`), which is his
    alternative with the letters as the parts. The table of the
    longest parts, half its size and equal in result, is built with
    the device tables of item 1.5. What goes now is the spelling
    rebuilt from the end-marked pairs and the greatest length
    (`WordIdentity.read_parts`, `assemble_pairs`), which cannot stand
    without end marks: its callers read the stored table, and a code
    that is no stored form has no spelling.
46. *A string read as a word it is not is taught when it is observed*
    (Alec, §19.64, point 3). Nothing is added to a form's code or to
    the match for it: no length that has to agree, and no match at
    every place. The teaching is an idempotent round over cases
    gathered from the training harness's logs. What is admitted is
    the part that discriminates the string from the word, and not the
    string (Alec, §19.65: "We don't want to admit Felixx, we want to
    admit the part that discriminates it from Felix: "xx""). *Mine:*
    the part is the shortest run that the string holds and the word
    does not, rule 43's own step; the string gets no row; once the
    part exists, the parts found in the string are no longer exactly
    the word's, so it is not read as the word wherever a word is the
    form whose parts are exactly those found (rule 44). Admitting a
    part that exists changes nothing. The store's side is this item's
    (certificate 28). The harness's log of a word put back as another
    word, and the round that gathers the cases, are item 1.6's.

**C. Order zero.**

7. *Combination within one reading* (Alec, §18.24, rule 22; §19.17;
   ""Pi on the gaps". You mean pi over the individual percept dims
   instead of pooling? Good."). For each lane, the vectors of the
   active percepts are combined into one vector as wide as the
   perceptual space, and the Sigma reads the for vector less the
   against vector. The combination is, for each dimension, a Pi over
   the percepts of the reading, taken on one less each value:
   $x_i = 1 - \prod_p (1 - v_{p,i})^{b_p}$, with $b_p$ the Pi's value
   for percept $p$ on that lane and $v_p$ its full vector. `.where`
   and `.when` are read unsigned; since item 1.8 they are stored so
   (1.8 §59.62), and a reader of an angle takes $2x - 1$. *Mine, from
   round 2 (§23.2.1), and put to Alec:* the `.where` and `.when`
   coordinates of a reading are not combined over its percepts. A
   reading has one place and one time, its own, read at the degree
   that it is present. Combined over the percepts they say how many
   percepts there are and no longer where (0 of 48 places read in the
   trial).
8. *The concepts of order zero* (Alec, §18.24, rule 23): the Sigma of
   rule 3 on $x$, with $U$ dense, of the perceptual width by the
   concepts. It replaces the fold of `ConceptualSpace._compose_order0`,
   the readout `IndexedSigmaConceptsFromPercepts`, and item 1.8's
   stand-in for one signed weight (`SparseLayer._fold_weights`) at
   order zero.

8b. *A concept made at order zero starts from its members* (*mine*):
   a new column of $U$, the sum of its members' vectors, normalized,
   which the network then learns; under rule 21 a new concept has one
   member, its own row. What promotion, `PerceptCollections.collect`
   and the lessons write today (an edge in the feature matrix,
   `add_concept_feature`) becomes such a column. The Pi of order zero
   holds only the mereology's entries, a percept over its parts and
   wholes; no concept writes into it, and it stays full rank.

**D. Above order zero.**

9. *Symbols are split from concepts and have no dimensions* (Alec,
   §18.22, rules 14, 15 and 20). Separate tensors; a symbol has no
   vector, place, form or trust content read as a direction; the
   shared row of a concept and its symbol is given up.
10. *A symbol is two unsigned rows, one for each lane, and brings one
    concept* (Alec, §18.25; rules 2 and 21). It is made when MTM
    nominates it (rule 20), or by `interpret` for a meta concept, and
    not at a concept's allocation. A conjunction is promoted as a new
    symbol, a Pi row over its members, under a new concept, a Sigma
    over that symbol.
11. *The Sigma above order zero* (Alec, §18.24, rule 24, with
    §18.25): it reads each symbol's pair combined into one signed
    input. No vector and no pooling. *Mine:* stored sparse.
12. *Sigma and Pi at every order* (Alec, §18.24, rule 25). The least
    and the greatest of `SparseLayer.fold_presence` go above order
    zero too, and item 1.8's `SparseLayer._fold_weights` with them.
13. *Place and time* (Alec, §18.24, rule 26). A percept's vector
    carries `.where` and `.when`; a symbol stores neither; its
    `.where` is written when STM reads it. Where a product reads them
    they are taken unsigned (rule 7).
14. *Between conceptual orders, only activation* (Alec, §18.22, rule
    16): a pull within one order does not pass through a symbol.

**E. Learning and the way back.**

15. *Membership is learned under a penalty* (Alec, §18.24, rule 29;
    rule 22). *Mine:* §18.3's penalty, measured first, on the Pi's
    entries and on the Sigma's entries above order zero. The Sigma of
    order zero is dense and is not penalized. Above order zero a
    promotion writes an edge at one into capacity allocated
    beforehand; at order zero it writes a column of $U$ (rule 8b).
    The penalty thins the values of entries that exist; it does not
    make a matrix sparse (§18.9).
16. *Likeness of codes pairs nothing* (*mine*, from §18.24, rule 28).
    A row enters a set, or two rows are taken as neighbours, by the
    mereology or by a promotion, never by the likeness of their
    codes. The readers that decide by likeness today are §19.4, list
    1, and the table of §19.21.
17. *The way back* (Alec, §18.24, rule 30; §19.24). The Pi inverts
    exactly; the Sigma returns the combined vector by least squares
    where there are enough concepts, and not otherwise. A signed
    value on the way back is replaced by its two unsigned parts
    before a percept or a symbol holds it, as on the way up: -.5 is
    .5 on the against lane and nothing on the for lane. Teaching does
    not assume an exact inverse; where it does today is §19.4, list
    2.

**F. The learned maps of System 1, and storage.**

18. *The stacks are the same structure* (Alec, §19.10; rule 29). A
    fused pair, `SigmaPiLayer` (new: no class of that name is in the
    source), and `SigmaPiStack` (`expect`, the read from a field to a
    point, and the marker of `bin/Language.py`) are pairs of rules 2
    and 3, a Pi and then a Sigma whose entries are for or Not, with
    no way back, fused so that a stack is captured as one graph.
    Today `bin/SigmaPiStack.py` uses `PiLayer`'s chart of odds and a
    linear `SigmaLayer` clipped to 0 to 1. Rules 6 and 7 of §18.5
    stand for the rest (the keys, the pair count, the seed, the last
    Sigma starting at zero, the cost of an MLP layer). `expectLayers`
    and `ideaLayers` stay at 1, and `markerLayers` at 2. *Mine:* the
    stacks read concepts, which are scalars (§19.19); and the clip to
    0 to 1 goes with the old arithmetic, since rule 3's Sigma gives
    one signed number in -1 to 1 (§19.64, point 6).
19. *The matrices are tensors on the device* (*mine*): the Pi and the
    Sigma above order zero are stored as compressed rows, not as
    dictionaries of edges (`SparseLayer._index`), and grow into
    capacity allocated beforehand. They can be read by row and by
    column (rule 28), with an index of columns kept beside the rows.

**G. Rows and concepts** (§19.10).

20. *MTM nominates every row of the Pi* (Alec). A basic percept at
    order zero, and a symbol at every order above it, becomes a row
    only when MTM nominates it on recurrence (`concept_mint_count`);
    nothing becomes a row on its first reading. *Mine:* the tower of
    parts and wholes below the basic percepts is perception and is
    made on first sight, as now. This replaces the basic percept
    reserved at a word's first reading (`BasicPercepts.reserve` in
    the word loop of `bin/Models.py`), the lessons' declared
    admission (`Corpus.admit_training_forms`,
    `ConceptLessons.admit_native_forms`), and symbols made at a
    concept's allocation.
    - *A new word* is perceived at once, but `interpret` cannot find
      it until its basic percept is nominated; its concept comes with
      that row (Alec, "A,b,c,d, yes").
    - *Not nominated* (Alec): the meta concept of a word and its
      object, which `interpret` makes, with its structure, when it
      interprets the word and the meta concept does not yet exist;
      the blank object, below; and the grammar's declared concepts,
      the 72 arguments and 27 operator classes of §19.9, each
      declared with its row at construction.
    - *The 12 frozen thought predicates* of §19.9 stay columns of $U$
      at zero, frozen, read by thought as references and never by
      perception (Alec, "E nice!").
    - *The blank object* (Alec). For a word with no object,
      `interpret` makes a new concept with no members, together with
      the meta concept; today MTM holds such a word as a pending name
      (`InterpretLayer.reading`, `pool.named`). Its vector starts at
      zero ("Pure zero is even better, since magnitude is certainty
      in that space"); a read that normalizes a concept vector must
      not divide by its zero length. *Mine:* it is one order above
      its word.
21. *One concept for each row* (Alec). Each nominated row brings one
    concept, so every order's Sigma is square: as many concepts as
    inputs, with no halving between orders. A concept starts reading
    only its own row, at one; at order zero that is the column of $U$
    given by the row's own vector, normalized. Concepts have no
    nomination or promotion of their own: MTM's promotions of
    classes, identities and percept collections become nominations of
    rows. The one writer of a concept's further inputs stays: a kept
    reading adds the word that filled an argument to that argument
    (`ConceptHierarchy.observe_member`), as an entry on the word's
    symbol once MTM has nominated it.
22. *Both thinned* (Alec). Rule 15's penalty is on the Sigma's
    entries above order zero as well as on the Pi's; §19.13 shows
    that with the Pi alone penalized what is learned cannot be read
    from the rows. The structure of a meta concept is never thinned.
    *Mine:* a definition between concepts ("balloons are rubber", an
    edge taught from its `LTM1` part row by `DefinitionFold`) stays
    under the penalty, held by its witness's cost.

**H. The vertical spread** (§19.11).

23. *A fan up and a share down* (Alec), on the structure's own
    matrices, Pi and Sigma alike, by the size of their weights. Up: a
    concept's or a symbol's value is split among the rows that read
    it, each by its weight's share of all the weights it is read with
    (its fan). Down: a row's value is split among its inputs, each by
    its weight's share of the row. What a row receives is summed over
    all its parents and children, at most one; a Not entry passes to
    the other lane. At order zero the way down goes on into the first
    layer (rule 30); the combined vector of order zero's Sigma is not
    passed (rule 17).
24. *Siblings get less than the concept that caused them* (Alec). A
    concept present at one gives every other concept the spread
    reaches less than one.
25. *One pass, written to the expected field at every order, where
    the grammar calls for it* (Alec, §19.22). Up once, then down
    once, from order zero's expectation (`expect`) and the observed
    field. It writes `_expected_knowing` at every order, where
    `ConceptSpread.open_prediction` writes order zero alone today. It
    is what `intuit` runs after it deposits expect's field; expect
    itself is the map at order zero alone, and a model with no
    symbols never runs the spread over the orders. *Mine:* it also
    runs at the end of the read, before comprehension takes its
    leaves' objects (1.8 §68.26), so that
    `InterpretLayer.preferred_objects` reads what the sentence's
    words primed.
26. *Not strictly ramified* (Alec). A symbol of order $k$ reads the
    concepts of every order below $k$. A meta concept is one order
    above the higher of its word and its object and reaches both in
    one step, with no symbol between. *Mine:* it holds for every
    symbol, not only meta concepts.
27. *Nothing new to learn* (Alec). The spread reads the structure's
    matrices, adds no parameter and no teacher, writes only the
    expected field, and changes no value of the evidence.
30. *The spread reaches the first layer, and the way back reads what
    it expects* (Alec's request of §19.23; the form is *mine*). The
    fan and the share run on the Pi of order zero, the mereology's
    own entries: a part that is present raises the rows that read it,
    and a row gives a share to each of its parts, so a part that is
    missing receives from the rows its neighbours raised. It writes
    expected percepts and changes no evidence; a row still requires
    its members (rule 2). At the reconstruction, where a look's
    percepts were withheld the way back reads the expected ones in
    their place. A concept reaches its row directly (rule 21).

**I. Thought reads the structure** (§19.12).

28. *Thought reads the Sigma and the Pi* (Alec). `parts(c)` gives the
    entries of c's row of the Pi: the symbols below c, at every lower
    order (rule 26), each with its weight. `wholes(c)` gives the rows
    of the Pi above c that have c as an entry, each with its weight.
    Of the Sigma, thought can read what a concept reads (its own row,
    and the inputs `observe_member` adds), which concepts read a
    given row, and which of those entries are Not entries. A result
    is a set of references with their weights, not a yes or no. A
    reference's words are those whose meta concepts join them to its
    concept; a reference with no word, such as a percept, is given as
    itself. *Mine:* an entry is read as present by a low threshold,
    which is a key, measured with rule 15's penalty; a read is serial
    and is charged to the work budget as the queries are today; it
    moves nothing.

**J. The order of learning** (§19.25).

31. *Words, objects, then their meta concepts* (Alec). A word
    generalization and an object generalization are each learned from
    their own occurrences, each a row that MTM nominates with its
    concept; the object's may be of another modality. An occurrence
    keeps its place and time, and a generalization holds neither.
    Then the meta concept of the two is learned from their consistent
    association. Nothing joins independent tokens: "there isnt a
    statistic that will magically produce identity across independent
    tokens". Two words are joined through what both are joined to.
    The clauses in this item's terms, and how it meets rule 20's meta
    concept and blank object, are in §19.25 and §19.27.

**K. The looks** (§19.28 to §19.64).

32. *A look is a `.where` that returns percept indices, and a Pi
    finds the spellings* (Alec, §19.28). A look is not made of the
    stem's word intervals: "If that is forced to words, then the
    lexer is doing the models job". Different spellings of a word
    ("caat", "cat") share most of their parts and wholes, so a row of
    the Pi in that neighbourhood finds them and returns the basic
    percept that corresponds to all of them. *Nothing more is
    specified for it* (Alec, §19.64, point 5): what a near spelling
    reads, and how wide a row's cone is, are what rules 2 and 15
    give, and they are reported. The contract of a look is 1.8 §74:
    its read is the request and then the Pi, with no reference to
    word intervals, and a look with the same request gives the same
    result whatever was looked at before it. The mereology is built
    once for the sentence, and the Pi makes the basic percepts "as
    something that changes in dependence on the .where supplied by
    the attend() mechanism". *Mine, a note from the prototype
    (§19.51):* a form the store does not know is read by its parts,
    and a row over parts tests containment: `the` is wholly present
    at `there`. Where the longer form is known, its own row is the
    larger of the two and is taken (rule 37). Where it is not known,
    the shorter word is read in its place until it is taught (rule
    46).
33. *The percept bank of a sentence* (Alec, §19.34, §19.40). At a
    sentence's start one lookup reads its percepts into one bank, on
    the device. For each event the bank holds its row, which is the
    what: a form's row where the store knows the form, and the rows
    of the form's parts (rule 43); its place, which is its start; its
    band of where wires (rule 47); its time; the token it belongs to;
    and its activity on two lanes. Nothing on a look's path calls
    `materialize()`, and nothing on it reads a span on the host. The
    first Pi is read only on the rows the lookup names. *Mine
    (§19.50):* the events are the lookup's rows and not bytes. The
    tokens are the stem's units until the lookup moves to the device
    (rule 44), with one addition: a known form that spans several
    units is an event of its own at its start (certificate 34's `ice
    cream`).
34. *A look is an activity over the bank* (Alec, §19.48, answer 2;
    §19.63, point 3). An event's activity in a look is its own, times
    its what match, times its where match: a percept is in the look
    to the degree that both its what and its where are salient. What
    is not active is absent on both lanes and is never evidence
    against (his rule for `non`, 2026-11-12). *Mine (§20.10; round 2,
    §23.2.1):* an event's activity in a look, before the where, is
    its share of the request capped at its own activity. The share
    already carries the activity, so the two are not multiplied. An
    event that two asked rows share is not counted twice. The where
    match is rule 47's, and multiplies a reading once (rule 39).
35. *The request* (Alec, §19.48, answers 1, 2 and 6; §19.49; §19.54;
    §19.57; §19.63, point 4). It is the output of `attend`: a mask
    over all concepts, on two lanes, and it is learned. It names a
    place in the tree of concepts, and the more particular the node,
    the fewer words the map selects. As proof of concept it is the
    node "words", the concept that words are included in. Selecting
    "words" presses against the space: that concept has a Not entry
    on the space character (rule 3), so a part that holds a space is
    lowered. The gradient of a look's selection lands on the request,
    on which concepts are asked for and on what produces that asking;
    and `attend` exposes to the error the network's weights that read
    the where. *Mine:* the mereology's entries of the first Pi and
    the weights that read the what take none of that gradient, so
    that perception keeps its one teacher. `attend`'s map starts at
    the request "words" alone. The node "words" is one concept that
    every word's concept is a member of, and its Not entry on the
    space is declared with it. A concept "objects" is declared beside
    it (Alec, §19.65: "Since we create the meta by hand, we might as
    well add the "objects" concept also"): one concept that every
    object's concept is a member of, an object being what `interpret`
    finds or makes for a word. *Open:* which rows of the source these
    two are (§19.29).
36. *The salience map* (Alec, §19.43; §19.48, answer 2; §19.53;
    §19.60; §19.64, point 1). The request is carried down through the
    matrices of the way up as a gradient of the asked concepts by the
    percepts of the sentence. It is taken from the open run, the
    first look, in which the whole sentence is read at once. Each
    Sigma is passed by its weights, which is before its squash, so
    that a concept already lit still returns its percepts. Each Pi is
    passed by the derivative of its product at the values of the open
    run. It lands on two domains: on each event's activity, the what,
    and on each event's band of where wires. *Mine (§19.46, §20.2):*
    it is computed by the transposes of the same sparse matrices with
    the Pi's local derivatives kept from the open run, and no second
    backward pass is made inside the forward. Where a token has
    several copies of one row its membership is the greatest of them,
    and each copy at the greatest takes the whole derivative. *Found in
    round 2 (§23.2.1):* where parts are present by degree the map
    cuts weak words, since a share goes with the presence of its row:
    a word present at .3 comes back at nothing. And a request that
    asks at one or above gets no gradient on what it asks, through the
    cap of rule 34.
37. *The returned set* (Alec, §19.45; §19.49, answer 3; §19.53). The
    way up is run again on the bank with the look's activity. What it
    gives, each concept at its token, is the returned set. The
    salience map selects the level, and the where selects the word.
    `interpret` takes the most active word concept, exactly: a
    selection, with no degree. Of forms that begin at one place the
    largest is the word (§19.30). Where several are still equally the
    most active, the request has identified no word: one is taken by
    a draw, and the draw is credited by the outcome of its path, as a
    departure of an explore path is (`explorePaths`,
    `SentenceCredit.WALKS`). *Mine:* largest is the longer extent, and it is taken at each
    place first, so that a tie is equals left at different places;
    where the set holds no word concept, nothing is interpreted; and,
    with one signed number for a concept, a word read at for 1 and
    against 0.0002 ("Felix", the decision carried from 1.8 §64.7) is
    just under one, is the most active at its place, and is
    interpreted (§19.49).
38. *`attend`* (Alec, §19.45; §19.48, answers 5 and 7; §19.49;
    §19.53, point 3; §19.62). One `attend` is: give the mask over
    concepts, carry it down as the salience map onto the what and the
    where, intersect, go up, and hand the returned set to the
    grammar. The acts on brackets go. No order is built in: reading
    the words of a sentence in order is taught, and a test of it
    gates the grammar tests. The reading lesson stays and is
    rewritten for requests in place of acts on brackets. `attend`
    asks for the word level in general and changes the location it
    asks for from the state of STM, so as to find the next available
    word. No place is given as a number and no step is taught: the
    trial of §20.8 is not built. Teaching begins with minimal
    sentences and grows a step at a time, so that a parse that fails
    differs from a shorter one that succeeds by one step. *Mine:*
    `attend` reads the context it reads today
    (`AttentionContext.context`: the understood prefix with its
    places and times, the heat of the spaces, the surprise). A form
    that was interpreted is lowered in the bank's activity over its
    extent, the `done` mark of today. Each word is composed as it
    arrives, as today (1.8 §60.1, rule 6). The operations declared
    for attention (`focus`, `unfocus`, the field's and, or and not)
    act on the request's two lanes where they are kept. *Open
    (§19.61):* what happens after a look that interprets nothing, and
    looks that are not the next word.
39. *Attention dims; it does not cut* (Alec, §19.36; §19.48, answer
    2; §19.63). What lies away from where the request looks is
    lowered by degree, by the radius and the magnitude of what asks,
    within the mereological field. There is no separate floor, and
    the key `attentionFloor` is not read by a look. The place row
    built for the purpose (§19.51, §20.5) goes with rule 47. *Mine
    (§20.5):* the where match is read at a form's place, its start,
    and applied once to the form's reading, on both lanes. It is not
    applied inside the product of a row's parts, where it cuts the
    word that was asked for.
40. *One gain field* (*mine*, from what §19.37 found for him: the
    gain field joins attention and expectation, and `attend` does not
    feed it). A look's activity is the `attention` of
    `Expectation.gain_field`, and the look's forecast (1.8 §62.3) is
    its `expected`. Its excess and omission are the surprise `attend`
    reads; `LookForecast.difference` goes.
41. *One where* (Alec, §19.48, answer 4; §19.63). The where is the
    place on the constant sinusoid, on wires of its own (rule 47).
    There is no second where for how a thing's parts stand to one
    another. The tests of spans made on the host go (`contained` in
    `read_percepts` and in `PartSpace.part_memberships`, `covered` in
    `Attention.narrowing_view`). `.where` stays on every reading.
    *Mine:* the order inside a form is in its code only as far as it
    tells known forms apart (rule 43), and a string that is no known
    form may read as one (rule 46). With the registry's capacity the
    short rung separates the words of a sentence and the long rung
    removes the short rung's repeats 256 places away; a capacity so
    small that both rungs have one period is not used for a look
    (§20.6).
42. *A literal* (Alec, §19.48). A word left uninterpreted may enter
    the tree as a literal, a sign and not a symbol, as if it had been
    quoted. It stands where a clause would. It enters only where the
    grammar has a rule that takes it, such as `sign` (`SignLayer`);
    where none does, the sentence is not well formed (rule 48).
47. *The where is a pathway of its own, on fixed wires* (Alec,
    §19.62, §19.63; §19.64, point 7).
    - The `.where` that is stamped onto percepts flows from one
      parameter that holds the constant sinusoid ("a pytorch param
      with the sinusoid written to it"). It is fixed: nothing of it
      is learned, and nothing is written into it as a state. Each
      percept has its own band, the wires of the code at its place.
    - The salience map (rule 36) lands on those wires, and what it
      lights there says where to look.
    - The combination is the intersection of what and where (rule
      34).
    - No location concept is made. "Concepts have the location to
      abstract over, and a gradient incentivizing them to do so, so
      it will learn location-concepts over the network, and attend()
      will expose that network gradient to the error."
    - *Mine:* the parameter has one row for each place of the
      registry (`WhereRegistry`), written from `WhereEncoding.encode`,
      and a band is its row, taken by the percept's place. It takes a
      gradient and is in no optimizer's groups: "The parameter itself
      must stay fixed if the sinusoid coding is to remain meaningful
      across all .where" (Alec, §19.66). It is the one where tensor
      of every space, a row for each address of the registry
      (§19.66). Concepts read
      the where through the weights they have on the where
      coordinates of the combined vector (rules 7 and 8). Those
      weights start at zero for a new concept, since a generalization
      holds no place (rule 31), and they are learned.
    - *Mine, the where match; tried in round 2 and built so
      (§23.2.1).* A reading's where is the band of the event at its
      start, weighted by that event's presence. An event's where
      share is the salience on its band
      taken with the wires' values: the sum over the band of the
      gradient times the value. The where match is that share
      referred to the sentence's most salient place and passed
      through the Pi's exponential. It is one at the most salient
      place and lower away from it, by the magnitude of what was
      asked; and it is one everywhere when nothing of where was
      asked, so that a request with no where part selects the level
      and leaves every place in (§19.53).
    - *Settled by round 2 (§23.2.1).* The concepts' weights on the
      where coordinates are enough: with them the word at a place is
      returned in 16 of 16 sentences, and reading in order was learned
      from every start, with no location concept made. A row of a Pi
      on the where wires has no share at its own place and is not
      built. Location concepts come of teaching location (rule 49).
      The fallback of §19.48 is not built.
48. *Silence* (Alec, §19.64, point 4; §19.53, point 3; 1.8 §68.89,
    §68.91). When the read of a sentence gives no well-formed
    formula, the sentence is written by force as an empty sentence
    that ends in NULL, by the rule `silence`, S → NULL, and the read
    ends there. The rule is in no grammar file and nothing chooses
    it: it is a fallback for a grammar that is broken. A read that
    gives no formula is an error, and the error is propagated.
    *Mine:* `silence` is a rule's layer registered by its name beside
    the others (`GRAMMAR_LAYER_CLASSES`), with no operand. It is
    applied where the closing finds that the sentence's trees did not
    reduce to one (`Models._stm_reduce_to_single_S`), in place of
    leaving the sentence with no row. The row it writes holds no word
    and no concept, and MTM takes nothing from it. What is put back
    for the sentence is NULL at every place, where the standalone
    kernel's head gives 0.5 today. The sentence's cost is booked, and
    that is how its error is propagated. A literal with no rule to
    take it (rule 42) ends so. *Open:* whether the sentence is tried
    again in the same pass ("we can try again"). The mining of such
    sentences by bisection is item 1.6's (1.8 §68.89).
49. *Location is taught, and the reconstruction teaches it* (Alec,
    §19.65). "Reading in order will create location concepts as long
    as we are teaching location." The bytes that are put back are
    built through a conjunction of the what pathway and the where
    pathway: a percept that is recreated has a location that is
    recreated with it, and that location places it in the out buffer.
    So the reconstruction's error is over both pathways. *Mine:* on
    the way back a concept gives each percept it recreates a band of
    where wires, through the same weights by which it reads the
    where. The band is matched against the code of every place, and
    the match is the percept's weight at that place of the out
    buffer. What is put back at a place is the sum over the recreated
    percepts of what each is, times its weight there. The error
    against the sentence's bytes at their places then reaches the
    weights that read the what and the weights that read the where.
    It takes the place of putting a percept back at the span its read
    had (`SentenceField.native_target`, `FieldReverse.reverse`,
    `Models.reverseReconstruct`; to be read again in round 6).
    *From round 2 (§23.2.1):* the match of a recreated band against
    the places is a softmax over the places of the out buffer; the
    product of §20.5 gives no gradient from rest. Taught beside the
    reading lesson it read held-out orders 6 of 6; alone, 4 of 6.
    *Open:* how it sits with rule 30, where the way back reads what
    the spread expects; and that a concept has one where for all its
    occurrences, so a word that occurs at several places is put back
    at one.

**Certificates.** A certificate of a mechanism must pass. A learning
certificate is recorded and not repaired (1.8 §43); those are marked
*learning*.

1. *The old classes are unchanged:* `SigmaLayer`, `PiLayer`,
   `PiLayer2` and `SigmaLayer2` give the same values before and
   after, on random inputs over the whole range.
2. *The Pi:* a row requires its members (§18.24, certificate 21);
   bounds over the whole range, rails included, with weights scaled
   far beyond their start (§18.6, certificate 4); the exact way back
   (atol 1e-6, §18.6, certificate 5); a row has no weight outside its
   own lane; a membership of exactly one passes its derivative, and
   one below the floor passes none.
3. *The Sigma:* each concept is one signed number in a reading and
   its two lanes are its parts, never both above zero (Alec's
   numbers, §19.24: .5 is for .5 and against 0; -.5 is for 0 and
   against .5); on the way back a value of -.5 reaches the percepts
   as .5 on the against lane and nothing on the for lane; no symbol
   and no percept holds a number below zero; bounded as in 2; it has
   no bias; an entry is for or Not, never both.
4. *The cycle:* the next Sigma reads one combined input, for less
   against, for each symbol; no symbol is made at a concept's
   allocation; a promotion is the nomination of a row, which brings
   its one concept.
5. *Order zero:* §18.24's certificates 15 (a new percept needs no new
   weight), 17 (concepts told apart, reported), 18 (the mereological
   order), 19 (the way back as far as it goes) and 20 (place and time
   weights, reported). The combined vector is one less the product of
   the gaps, lies between zero and one, and a percept at zero changes
   nothing.
6. *Symbols:* §18.22's certificates 11 and 12 (no tensor holds both a
   concept and a symbol; nothing reads a vector, place, form or trust
   content at a symbol; a Sigma above order zero is unchanged when
   the places of every other order are rotated).
7. *The stacks:* a fused `SigmaPiLayer` equals its two layers (atol
   1e-6); a stack is one graph; seconds against an MLP layer of the
   same size; depth (§18.6, certificates 3, 6, 7 and 8, on rule 18's
   structure). And the least and the greatest coordinate of the root
   ideas of the lesson's sixteen sentences are printed and lie within
   what the read to a point can give (§19.64, point 6).
8. *Likeness and the way back:* no reader of §19.4's lists 1 and 2,
   or of the tables of §19.21 and of this section, is left as it was,
   each changed as listed or reported.
9. *The standing gates and the lesson* (§18.24, certificate 16;
   §18.6, certificate 10): both stopping gates pass, XOR among them;
   the lesson's readings before and after, each order's occupancy,
   the Pi's kept edges for each row, the Not entries at each order,
   and one default sweep.
10. *Speed* (*mine*): the seconds of the lesson's sixteen sentences
    before and after, as §51 of the 1.8 specification reported them.
11. *No row on a first reading.* A new word read once has no basic
    percept and no concept, and `interpret` does not find it; after
    `concept_mint_count` readings MTM nominates its basic percept,
    its concept comes with it, and `interpret` finds it.
12. *Square.* At every order the concepts equal the rows in number,
    and a new concept reads only its own row, at one; at order zero
    its column of $U$ is its row's vector, normalized.
13. *Thinned.* Under the penalty, a Sigma entry above order zero that
    no cost supports goes to zero, as a Pi entry does; the Sigma of
    order zero is not penalized.
14. *Up.* A word found raises its meta concept by its share.
15. *Down.* A primed meta concept raises its object when the object
    is absent; the way back of rule 17, on the same case, does not.
16. *Rule 3 of 1.8 §55.1.* Finding a word raises its meta concepts
    and their objects; of a word's two objects with equal
    expectation, the one whose meta concept another word of the
    sentence primed is preferred at comprehension.
17. *Evidence unchanged.* With the spread on and off, every order's
    evidence is equal, and the parameters are the same.
18. *The blank object.* Interpreting a word with no object makes one
    blank object and one meta concept, and the blank object's vector
    is zero; interpreting it again makes nothing new.
19. *Siblings below the cause.* With one concept present at one,
    every other concept the spread reaches is below one; two children
    sharing five parents of equal weight give each other .5.
20. *One step across orders.* A symbol of order two reads concepts of
    orders zero and one; a meta concept over a word of order zero and
    an object of order one primes the object in one pass, with no
    symbol between.
21. *The structure is readable.* For a concept c, `parts(c)` and
    `wholes(c)` give the references and weights of c's row of the Pi
    and of the rows above that have c as an entry, with the words of
    those that have words, and the Not entries are told apart; the
    matrices, the activations and the rows are unchanged after the
    read.
22. *And, or and not.* A stack of rule 18 learns and, or and not over
    true, neither and false from random starts; and and or use no Not
    entry; with rule 22's penalty, the network whose present entries
    are set to one and the rest to zero gives the same answers.
23. *Pi, Not, Sigma.* (a) With no omission taught, the basic level
    has no Not entry, raising an input never lowers a concept, and no
    concept is against. (b) The basic level alone learns "A or B, not
    both" and "A alone, not A with X" from what was expected and what
    came, and after it X alone is expected absent. (c) The basic
    level alone learns XOR of two perceived words with no Not entry,
    given a row for the pair. (d) Above the basic level a Not entry
    is a symbol's pair read swapped.
24. *A blind spot is filled* (Alec, §19.23; the testpoint
    `TestBlindSpot`). With half of a known word's forms withheld, the
    reconstruction gives the whole word at its place; the word beside
    it is unchanged; a word never taught is not completed. The values
    of §19.23 are printed.
25. *A word and an object of another modality are tied by their
    consistent association* (Alec, §19.25; the testpoint
    `TestCrossModalTie`). The word alone and the stimulus alone each
    make a row and a concept, and no tie; presented together they
    make the meta concept; then each leaves the other expected; a
    word met with a different stimulus each time is tied to none; a
    second word met with the same stimulus is tied to the same
    object. The values of §19.25 are printed.
26. *A sequence is learned* (Alec, §19.25; the testpoint
    `TestBellSalivate`). After the first stimulus has been followed
    by the second, the first alone leaves the second expected, and
    its response through it; a stimulus never followed by the second
    does not. The values of §19.25 are printed.
27. *The structure as composed is valid* (Alec, §19.53: "Please add
    a test point to ensure the validity of the structure as
    composed"; the testpoint `TestStructureValid`). On the lesson's
    words, after a suffix lesson and after two forms have had to be
    told apart: (a) every part the store records for a form lies
    wholly inside the form's code; (b) every known form found inside
    a known form lies wholly inside it, read by the one match
    (`ConceptualMatch.match`): `is` in `this` reads 1, where it reads
    0.82 today; (c) the count of forms that read as wholly inside a
    form they are not inside is printed, with `cat` in `caat` named.
28. *A string that is no form* (rule 46). (a) With `felix` known and
    `felixx` not, the two have the same parts; what `felixx`, `caat`
    and `xileF` are read as is printed. (b) After the part `xx` is
    admitted, the parts of `felixx` hold one that `felix` lacks,
    `felix` still lies wholly inside them, and `felixx` is no form of
    the store. (c) Admitting the part again changes nothing: the
    store's rows, codes and keys are equal before and after.
29. *The way back of a form* (rule 45). Every form the lesson stores
    is put back exactly from the form alone; `read_parts` and
    `assemble_pairs` are named nowhere in `bin/`.
30. *The bank is read once* (rule 33). One lookup for a sentence; on
    a look's path no `materialize()` and no host read of a span,
    counted on the lesson's sentences.
31. *The salience map* (rule 36). (a) The transposes give what the
    framework's backward gives on the same local derivatives (atol
    1e-6). (b) A request for a word's concept returns it at its
    token; for a word that is absent, nothing; for "words", every
    word; and on the 304 requests of §19.52, exactly the asked words
    that are present. (c) Printed: the same requests where parts are
    present by degree, beside the plain gradient and beside the rows'
    entries alone.
32. *The largest, not a word inside it* (rule 37). A known word is
    not taken at the token of a longer known word that holds its
    letters: `is` at `This`, `here` at `There`, `the` at `there`.
33. *The where pathway* (rule 47). (a) The parameter equals
    `WhereEncoding`'s code at every place, before and after a
    training step, and is in no optimizer's groups. (b) A gradient
    from an asked concept that reads the where reaches a percept's
    band. (c) With a concept given weight on the where by the test,
    the request "words" with that concept returns the word at that
    place on each of the lesson's sixteen sentences. (d) With nothing
    of where asked, every word of the sentence is returned and none
    is lowered.
34. *"Words" presses against the space* (Alec, §19.57). With a form
    of two words known (`ice cream`), a request for "words" lowers
    the part that holds the space below the two words inside it, and
    without that request the larger form is the one that fits; on the
    lesson's sentences no part selected by "words" holds a space.
35. *A word among garbage letters* (Alec, §19.53; 1.8 §68.88) is
    returned, and it alone is said.
36. *The same request, the same look* (1.8 §74). A look with the same
    request gives the same result whatever was looked at before it.
37. *Dimming* (rule 39). An event away from where the request looks
    is lowered on both lanes and is never read as against;
    `attentionFloor` is read by no look.
38. *Reading* (rule 38). (a) A fresh model with `interpretBias` at 1
    reads every word of each of the lesson's sentences once: none
    twice, none left. (b) *Learning:* after the reading lesson the
    words are read in order, on how many of the sixteen. (c) The test
    of (b) gates the grammar tests, as the assessment's first rung
    gates its others.
39. *One gain field* (rule 40). The surprise `attend` reads is the
    excess and the omission of `Expectation.gain_field`, with the
    look's activity as its `attention` and the look's forecast as its
    `expected`; `LookForecast.difference` is named nowhere; where a
    look's activity is one inside one word and nothing elsewhere, the
    surprise is item 1.8's for that word.
40. *A literal* (rule 42) enters only under a rule that takes it;
    with no such rule the sentence ends in silence.
41. *Silence* (rule 48). A sentence whose read gives no formula is
    written as S → NULL; what is put back is NULL at every place; its
    cost is booked; no grammar file names the rule; a sentence that
    reduces is unchanged.
42. *A word not yet known* comes back under a request for the word
    whole with the whole lit and no word concept, and is given a
    name, as the "neither" of today is (§19.45).
43. *"Felix"* (1.8 §64.7; §19.49). A word read at for 1 and against
    0.0002 is interpreted.
44. *The cost of a look* (*mine*). The host reads and the seconds of
    the lesson's sixteen sentences beside item 1.8's, with
    certificate 10; and the operations of one look beside one way up
    (§20.4 counted 2.3).
45. *The reconstruction is over what and where* (rule 49). (a) A
    percept recreated with the band of another place is put back at
    that place, and the error says so. (b) The gradient of the
    reconstruction's error reaches the weights that read the where.
    (c) *Learning:* after the reconstruction is taught on the lesson's
    sentences, which concepts have weight on the where is printed.

**What stays.** The grammar's option at a look and `interpretBias`
(1.8 §60.1, rule 5b; `OperationSelectionLayer.interpret_at_look`); the
forecast of a look from what lies outside it (1.8 §62.3,
`LookForecast`); the open run (`FamiliarTier.open_sentence`,
`_sentence_open_fields`); `interpret` at one word; the reading lesson,
rewritten; the store's lookup, until item 1.5.

**What goes.** The bracket table and its acts
(`Attention.BracketTable`, `narrow_step`, `narrowing_view`,
`narrowing_mask`); the look's read of one bracket
(`SentenceLooks.perceive` as a test of containment); the old pass-back
(`Models._passback_scope_ps`, `_passback_scope_where`);
`LookForecast.difference`; the key `attentionFloor`, whose value
nothing uses today; the tests of spans on the host (`contained`,
`covered`); in the store, the end marks, the lengths, the mints at a
position (`WordIdentity._mint` as it is), `read_parts` and
`assemble_pairs`; and `SigmaPiStack`'s clip. `cs_percept_attribution`
keeps its other readers until rule 30's way back takes their place.
What a rule replaces is deleted, and not kept behind a key.

**Readers to adapt.** §19.4's four lists and the table of §19.21
stand. They were read on 2026-11-16 and 2026-11-17, and each round
takes its own again on its starting source. Added on 2026-11-19 (read
in the source at `b3d2e3ae89`):

| Where | Today | Under |
| --- | --- | --- |
| `WordIdentity.base_atoms`, and what calls it beside the store: `PartSpace.part_memberships`, `Omission.prepare` and `stage`, `AttentionLesson._prompt_need` | the pairs of `#word#` with the two ends marked, and the lengths 1 to n; a pair block of 64 coordinates with three bits for each atom, and 32 for the lengths | rule 43: the characters and the added parts |
| `WordIdentity._mint`, called from `admit` and `observe_nested` | for two forms with one code, a mint of three or four characters at a position (`cal@1`) | rule 43: the shortest run that one holds and the other does not, carried by every form that holds it |
| `WordIdentity.observe_nested` and `nested` | in a suffix lesson a stem's code and a marker's are joined into the longer form's, and every form is rekeyed | the codes need no joining; what the suffix lesson still records is reported |
| `WordIdentity.read_parts`, `assemble_pairs`, called by `InterpretLayer.lookup_word` and `FieldReverse.reverse` | a spelling from the end-marked pairs and the greatest length | rule 45: the stored table |
| `RadixLayer(word_bounded=True)`, `_promotable` | no promoted chunk crosses a space | a taught form may hold a space (rule 43); the promotion by frequency is unchanged |
| `WhereRegistry.encode`, `WhereEncoding.encode` and `stamp`; `PartSpace._radix_code_events`, `_embed_radix` | a place's code is computed when asked for; no parameter, no gradient | rule 47 |
| `SentenceLooks.Looks.run` and `perceive`; `Attention.narrow_step`, `narrowing_view`, `record_interpretation`; `OperationSelectionLayer.attend`; `FocusLayer`, `UnfocusLayer` | a look is an act on the bracket table, and its read is one bracket | rules 33 to 39 |
| `AttentionContext.context` and `project` | the context is mapped to acts on brackets, from a map that starts at zero | the context is mapped to the request (rules 35 and 38) |
| `AttentionLesson.ReadingTargets`, `reading_lesson` | teaches the fewest acts on brackets to the next word | teaches the request (rule 38) |
| `Expectation.gain_field`'s callers (`WordArrivals.step`, `SymbolField.arrive`, `SymbolRead.gains`, `Meaning.form`); `LookForecast.difference` | no look feeds the gain field | rule 40 |
| `Models._stm_reduce_to_single_S`, `_program_entries`, `_commit_sentence` | a sentence that does not reduce has no row | rule 48 |
| `SigmaPiStack._field_value`; `IdeaRead.complete` | every Sigma's result is clipped to 0 to 1 | rule 18 |

**What is mine, in one list, for Alec to overrule.** From §19.21: the
names `PiLayer3` and `SigmaLayer3`; each percept or symbol as two
unsigned rows; a row reading its own lane only; the Sigma with no
bias; the answer of a Sigma read as its signed number; sparse storage
above order zero; the penalty of §18.3; likeness pairs nothing (rule
16); the matrices as device tensors with an index of columns (rule
19); the stacks reading concepts as pairs of the same structure (rule
18); the blank object one order above its word; the spread run at the
open pass and before comprehension (rule 25); the threshold key and
the work budget of rule 28; the speed certificate. Added by this
section:

1. The derivative at a membership of one, in `PiLayer3` alone (rule
   2).
2. The part added for two forms is the shortest run that one holds
   and the other does not; the code is held as the list and its fold
   in today's width (rule 43).
3. The lengths go from a form's code (rule 43).
4. The lookup moves to the device in item 1.5, and the table of the
   longest parts with it (rules 44 and 45).
5. The part admitted for a string read as a word it is not is the
   shortest run the string holds and the word does not; the harness's
   part is item 1.6's (rule 46).
6. The bank's tokens are the stem's units, with a form that spans
   several as an event of its own (rule 33).
7. The node "words" is one concept that every word's concept is a
   member of, with its Not entry on the space declared; an object, for
   "objects", is what `interpret` finds or makes for a word;
   `attend`'s map starts at the request "words" (rule 35).
8. Of a look's error, only the request and the weights that read the
   where are taught (rule 35).
9. The salience map by the transposes, with the kept derivatives
   (rule 36).
10. Largest is the longer extent; nothing is interpreted from an
    empty set; "Felix" is read (rule 37).
11. `attend` reads today's context; the `done` mark is over a form's
    extent; a word is composed as it arrives (rule 38).
12. The where match is applied once to a form's reading (rule 39).
13. One gain field (rule 40).
14. The parameter of the where: one row for each place, in no
    optimizer's groups; and the arithmetic of the where match (rule
    47).
15. `silence` as a registered layer applied at the closing; the row
    it writes; NULL put back; the cost booked (rule 48).
16. The form of the conjunction in the reconstruction: a recreated
    band matched against every place's code, as a percept's weight in
    the out buffer (rule 49).

**Open, with what is built meanwhile.**

1. *The where match, and whether location is learned with no location
   concept made* (rules 47 and 49). *Closed by round 2 (§23.2.1):* the
   recommended form is built, and reading in order was learned from
   every start, with the reading lesson and the reconstruction as the
   teachers. What is left is whether it holds in the model, which
   round 6 reports.
2. *The salience map where parts are present by degree* (rule 36).
   Round 2 found that it cuts weak words. The map by weights is
   built, as Alec decided; whether a weak word should come back is
   his.
2b. *The where and the when of a reading* (rule 7): not combined over
   its percepts. Built so in round 3, put to Alec.
3. *Which row "words" is* (rule 35; §19.29). Built as one declared
   concept.
4. *What becomes of the 32 coordinates of the length block* (rule
   43). They stay empty.
5. *After a look that interprets nothing, and looks that are not the
   next word* (rule 38; §19.61). Built: the request moves because the
   state of STM and the bank's activity moved; a look that changes
   neither ends the read of the sentence.
6. *Whether a sentence that ends in silence is tried again* (rule
   48). Built: it is not; its cost is booked once.
7. *Whether `parts` and `wholes` read the matrices alone,* with
   LTM1's part table kept as the evidence for `isPart`'s proofs
   (§19.12; Alec: "We can spec it later"). Built so.
8. *On graded readings* the layers give the product logic and the
   grammar's field operators the least and the greatest (§19.13).
   They are left to differ.
9. *Whether `data/shamatha.grammar` keeps its `not` rule* (§19.15).
   It is kept.
10. *Whether the basic level's against lane is fed by perceived
    opposites* (1.8 §57.6) as well as by omissions. It keeps what
    perception gives.
11. *What becomes of the attention walk's own pair* once the one
    Sigma's value is the pair (§19.24). It goes with the fold that
    reads it.
12. *How rule 31 meets rule 20's blank object* (§19.27: "better never
    to have the blank at all"; both trained, the blank removed only
    if the two vectors come to overlap). Built: the blank is made as
    rule 20 says; what removing is, beside concepts kept forever, is
    not built.
13. *What holds a succession,* the bell and then the food, across two
    sentences (§19.25). Built: a row that MTM nominates for the two
    when the succession recurs, read by rule 30's spread.

**For the documentation pass** that follows the item's last round.
The table of §19.21 stands. Added:

| File | What it gains | From |
| --- | --- | --- |
| `doc/Mereology.md`, `doc/Spaces.md` | a form's code as the set of its parts: no end marks, the space as a character, a part only where two forms would be the same; exact parthood between forms; the where on fixed wires from one parameter | rules 43 and 47 |
| `doc/Language.md` | the looks: the bank, the request, the salience map, the returned set, `attend`; a literal; silence, in place of "outputs nothing" | rules 33 to 42 and 48 |
| `doc/Architecture.md` | the two pathways, what and where, beside the two-streams account and feature integration (§19.63), in "Biological plausibility" | rule 47 |
| `doc/Params.md` | `attentionFloor` gone | rule 39 |
| `doc/FutureWork.md` | the lookup on the device by the set of parts as an address; the teaching round over the harness's logs | rules 44 and 46 |

## 22. Item 1.7: the text for the implementers, in rounds (Claude, 2026-11-19; it replaces §19.8)

Alec, 2026-11-19: "Write the item's text for the implementers as one
hand-off in that file (it replaces 19.8), in rounds, each with its
targeted tests and its certificates, the test points of 19.53 and
19.57 among them. One default sweep at the item's push round; no slow
tier." And: "Start the implementers in Opus, in the same checkout, and
review each return against the source before the next round goes."

**The rounds.**

| Round | What it builds | Rules of §21 | Certificates | Returns as |
| --- | --- | --- | --- | --- |
| 1 | the two layers, the stacks, and the code of a form | 1 to 3, 18, 29; 5, 43, 45, 46 | 1, 2, 3, 7, 22; 27, 28, 29 | §23.1 |
| 2 | the looks, tried in tests only | 33 to 39, 47, 49 | reported; 31, 33, 34 and 35 on the trial's own bank | §23.2 |
| 3 | order zero, and rows by nomination | 6, 7, 8, 8b, 19, 20, 21 | 5, 11, 12, 23 (a) to (c) | §23.3 |
| 4 | symbols and the orders above | 4, 9 to 14, 19, 22; the meta concept and the blank object | 4, 6, 13, 18, 23 (d) | §23.4 |
| 5 | the vertical spread, the order of learning, thought's reads | 23 to 28, 30, 31 | 14 to 17, 19, 20, 21, 24, 25, 26 | §23.5 |
| 6 | the looks, in the model | 32 to 42, 47, 48, 49 | 30 to 45 | §23.6 |
| 7 | the readers, the measurements and the sweep | 16, 17 | 8, 9, 10 | §23.7 |

- Each round is carried by one implementer, Claude on Opus standing in
  for Codex, in the one checkout.
- Round 2 changes no file of `bin/` and runs beside round 1. Every
  other round follows Claude's review of the one before it.
- The test points Alec asked for: the structure as composed is valid
  (§19.53; certificate 27, round 1); a word among garbage letters
  (§19.53; certificate 35, rounds 2 and 6); "words" lowers a form
  that holds a space (§19.57; certificate 34, rounds 2 and 6); a
  blind spot is filled, a word tied to an object of another modality,
  and a sequence learned (§19.23, §19.25; certificates 24 to 26,
  round 5).
- The item's one default sweep is round 7's. Nothing runs under
  `RUN_SLOW` in any round.
- A round's text is amended in place where an earlier return changes
  it, with the date. Round 6's waits for round 2's return.
- *Amended 2026-11-19, after §19.65:* rule 46 and certificate 28 (a
  part is admitted, not the string; round 1), the declared concepts
  "words" and "objects" (rounds 3 and 4), step 6b of round 2, and
  rule 49 with certificate 45 (round 6). The implementers of rounds 1
  and 2, already running, were told.
- *Checked by Claude, 2026-11-19, at `b3d2e3ae89`:* the names of code
  in §21 and in these texts are in the source, but for those said to
  be new; both stopping gates exist; and
  `test/attention_thinking_certificate.py lesson` runs the lesson's
  sixteen sentences.

**The standing rules, for every round.**

~~~
Item 1.7 of BasicModel: the Sigma and Pi layers, the perceptual basis
and the looks. You build one round of it. The rules are §21 of
doc/plans/2026-11-14-sigma-pi-layers.md. Read §21 whole first, then
these standing rules and your round in §22, then the sections your
round names. §19.64 holds Alec's last answers. Where §21 and an
earlier section differ, §21 stands. The file is long: find a section
by its heading and read that part.

1. Work in /Users/arogers/github/WikiOracle/basicmodel on main, in
   this checkout. Commit locally and often, with explicit paths
   (never "git add -A" and never "git commit -a": files that are not
   yours may be in the tree). Never push, never force-push, never
   rewrite history. If git reports a lock, wait and try again:
   another round may be committing.
2. Do not edit SigmaLayer, PiLayer, PiLayer2 or SigmaLayer2.
3. A rule of §21 marked "(Alec, ...)" is his decision. It is not
   overturned in the source. Where you would recommend otherwise,
   build the path that keeps it, write your proposal, and go on.
4. A question is not a stop (1.8 specification, §75). Where a rule
   marked "mine" or "open" cannot be built as written, or two ways
   differ and §21 does not choose: write the question, the ways you
   see and the one you recommend; build the one you recommend in a
   commit of its own, marked provisional in the return; go on.
5. A certificate of a mechanism must pass. A learning certificate is
   recorded and not repaired. Both stopping gates pass at the end of
   every round. To obtain a pass you change no bound, learning rate,
   seed, capacity or presentation count; you weaken or remove no
   validation; you add no skip and no expected failure; and no test
   fixes a seed so that an assertion holds. A fixture that no longer
   fits is given more room at construction, with its inputs and
   assertions kept; nothing is resized during a run.
6. What a rule replaces is deleted, with its tests ported. It is not
   kept behind a key or as a second path. A ported test is listed
   with its old and its new assertion. No test is retired without
   naming what replaces it.
7. No language is named or given a special case in the code.
   Comments and docstrings say what the code does; they do not tell
   this item's history or name its rounds.
8. Tests. Run targeted tests only: your round's certificates, every
   test file that names something your round changed (find them by
   search; the ones known are listed with the round), and both
   stopping gates,
     test/test_mm_xor.py::TestMMXorConvergence::test_convergence
     test/test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class
   Run a file as
     BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest test/<file>.py
   with RUN_SLOW and RUN_TRAINED unset. No default sweep before round
   7, and nothing under RUN_SLOW in any round. A test that was not
   run is no evidence: name every file you ran, with its counts and
   its seconds.
9. A red case or a stop that this text does not name is yours to
   diagnose to its producer: measure, bisect, name the function, give
   the evidence.
10. doc/*.md is not edited in these rounds. The documentation pass
    follows the item.
11. The return is a section of the specification, §23.N for round N,
    committed: what was built, by file and function; each certificate
    with its test's name and its result; each choice taken on your
    own recommendation, with its commit; each test ported, with its
    old and its new assertion; the test files run; the commits; what
    surprised you; and what you would change in §21. Numbers are
    stated in the text, and receipts are not linked. Write it at the
    end of the file, reading the file's end again just before you
    write, since another round may have added a section. Keep notes
    for it as you go. Then stop: Claude reviews the return against
    the source before the next round goes.
~~~

### 22.1 Round 1: the two layers, the stacks, and the code of a form

~~~
Round 1 of item 1.7. Rules 1 to 3, 18 and 29 of §21 for the layers
and the stacks; rules 5, 43, 45 and 46 for the code of a form.
Certificates 1, 2, 3, 7 and 22; 27, 28 and 29. The two parts do not
depend on each other. Build A first.

A. The layers and the stacks.

1. In bin/Layers.py add PiLayer3 (rule 2) and SigmaLayer3 (rule 3).
   They are not wired into the model in this round. PiLayer3 gives
   PiLayer2's values on PiLayer2's domain; its derivative passes at
   a membership of exactly one, and is nothing below the floor.
2. Add the fused pair SigmaPiLayer (rule 18): a Pi and then a Sigma
   whose entries are for or Not, with no way back. Of §18.5, rules 5
   to 7, what stands is: no copy of the activation, no gate, no
   choice in Python on a tensor, one graph, no reverse, the last
   Sigma of a stack zero at construction, the keys, the pair count
   and the seed. Read §19.13 and §19.19 for a stack's pair: its rows
   are free of the diagonal, read their own lane, have weights at
   zero or more and no bias, and it reads concepts, which are
   scalars.
3. Rebuild SigmaPiStack (bin/SigmaPiStack.py) as a row of
   SigmaPiLayer. Its clip, _field_value, goes. Its users keep their
   keys: the expect stack and the idea stack of the conceptual space
   (where bin/Spaces.py reads expectLayers and ideaLayers;
   ConceptSpread.expected, IdeaRead.complete) and the marker
   (_MarkerKernel in bin/Language.py, markerLayers). Where a user
   reads two unsigned halves as a difference, or assumes a result in
   0 to 1, say what it reads now.
4. Certificates 1, 2 and 3 in a new test/test_item17_layers.py;
   certificates 7 and 22 in a new test/test_item17_stacks.py. For
   certificate 7's last clause, print the least and the greatest
   coordinate of record.meaning_root where IdeaRead.complete takes
   it, over the lesson's sixteen sentences (the first stage of
   data/corpus-lessons.json), beside the range the read can give.
   For certificate 22 read §19.13 and §19.15; the penalty is §18.3's.
5. The stacks' learning changes with their arithmetic. The tests
   that hold a learned value of expect, of the idea read or of the
   marker are run and listed with their values before and after. A
   learning certificate that no longer holds is recorded with its
   numbers and not repaired.

B. The code of a form.

6. In bin/WordIdentity.py a form's code is the set of its parts
   (rule 43). base_atoms gives the form's characters, the space
   among them, with no end marks and no lengths; atom_code and form
   follow. WordIdentity keeps the set of the parts that have been
   added, saved and loaded with the rest of its state. admit gives a
   form every added part that lies inside it.
7. Replace _mint. Where a form that is admitted has the same parts
   as a form already held, the shortest run of characters that one
   of the two holds and the other does not becomes a part. Every
   form held that contains that run takes it and is rekeyed, and no
   stale key or trie entry is left behind (admit leaves one after a
   mint today; only observe_nested rebuilds the trie). Where two
   forms' rows of fixed width are still equal, the part's bits are
   raised as _mint raises a mint's today. The layout's widths do not
   change (resolve_identity_layout): the 32 coordinates of the
   length block stay, and hold nothing for a form.
8. The callers of base_atoms and form outside the store follow:
   PartSpace.part_memberships (bin/Spaces.py), prepare and stage of
   bin/Omission.py, _prompt_need of bin/AttentionLesson.py. Find any
   other by search.
9. observe_nested and nested: the codes need no joining now. Keep
   what the suffix lesson (nested_lesson) records that something
   else reads; remove the joining of codes and the rekey of every
   form that it caused; say what is left and who reads it.
10. read_parts and assemble_pairs go (rule 45). Their callers,
    InterpretLayer.lookup_word and reverse of bin/FieldReverse.py,
    read the stored table (RadixLayer.inverse_table). Report what
    each did for a code that is no stored form, and what it does
    now.
11. A taught form may hold a space. Find what refuses one today
    (RadixLayer's word_bounded and _promotable, the isalpha return
    of observe_nested, the stem's units) and make the least change
    that lets admit(b'ice cream') give a form whose code holds the
    codes of ice, of cream and of the space. Promotion by frequency
    is unchanged. That the read of a sentence finds such a form is
    round 6's.
12. Certificate 27 is a testpoint, TestStructureValid, in
    test/test_testpoint.py, written as that file's classes are: it
    prints its values. Its words: the words of the lesson's first
    stage, cats by a suffix lesson, and this, his, is, at, cat, caat,
    act, here, there, the. Part (b) reads ConceptualMatch.match.
    Certificates 28 and 29 in a new test/test_item17_forms.py.
13. Port the tests that pin today's layout. Known:
    test_operators_round3a.py, test_hash_order_producers.py,
    test_item55_mechanisms.py, test_form_subspaces.py,
    test_percept_transitivity.py, test_whole_path_repairs.py,
    test_stream_smoke.py, test_word_admission.py,
    test_dual_towers.py, test_serial_object_meta.py,
    test_aligned_fold_binding.py, test_reconstruction_roundtrip.py,
    test_word_store.py, test_meronomy.py, test_percept_store.py,
    test_basicmodel.py, test_partspace_dynamic_capacity.py. Find the
    rest by search.
14. Measure and report, with no assertion: for the first 20,000
    words of /usr/share/dict/words (lower case, letters only, 2 to
    12 letters) admitted to the real store, the parts added, by
    their length; the forms whose row of fixed width is equal to
    another form's; and the seconds. Stop it at ten minutes and
    report how far it came.

Run: the new test files; every test file that names what you
changed; test/test_item18_idea_read.py; both stopping gates; and the
lesson's sixteen sentences once before your first edit and once
after your last, as §66.14 of the 1.8 specification ran them
(PYTHONPATH=bin:test, test/attention_thinking_certificate.py lesson
--output <a new folder under output/>, with no --compare). Values
change in this round by design: report what differs in the readings
(the words read, the trees) and the seconds, not equality.
Return as §23.1.
~~~

### 22.2 Round 2: the looks, tried in tests only

~~~
Round 2 of item 1.7: a trial of the looks, in tests only. Rules 33
to 39, 47 and 49 of §21. Read §19.45, §19.48, §19.49, §19.53, §19.60 to
§19.64 and §20, and test/looks_prototype.py, which this extends.

It changes no file of bin/ and no existing test but the prototype's
own three files (test/looks_prototype.py,
test/looks_prototype_report.py, test/test_item17_looks_prototype.py);
new files go beside them. It decides nothing and is not wired into
the model. Another implementer is editing bin/ in this checkout while
you work (round 1). If an import of bin/ fails for a reason outside
your files, wait a few minutes and run again; do not edit bin/.

What it is for. Round 6 builds the looks in the model. Three things
are not settled, and this round measures them first: how the
salience map behaves where parts are present by degree; what the
where match is; and whether a request learns to read the words of a
sentence in order when no location concept is made.

1. The bank, as rule 33 has it, for the lesson's sixteen sentences
   (LESSON in test/looks_prototype.py), the three of §19.52 ("Fe li
   x is here.", "xileF is here.", "The cat is there.") and one
   sentence with a known form of two words (ice cream). For each
   token: its form's row where the form is known, and the rows of
   its parts as rule 43 codes a form, which are its characters and
   the parts that tell two forms of the vocabulary apart, with no
   end marks. Build that code in the test; do not read it from the
   store, which round 1 is changing. A known form that spans two
   tokens is an event of its own. An event's place is its start. Its
   band of where wires is its row of one parameter that holds the
   code of Spaces.WhereEncoding at the registry's capacity; the
   parameter takes a gradient and is given to no optimizer.
2. The way up: a first Pi of rule 2 whose rows read their own form
   and their parts; the combination of rule 7, with the where
   coordinates among its dimensions; the Sigma of rule 3, dense,
   with every concept's weights on the where coordinates at zero at
   the start; and the node "words" as rule 35 has it, with a Not
   entry on the space.
3. The salience map of rule 36: by the transposes, with the
   derivatives kept from the open run; and by the framework's
   backward on a graph in which each Sigma is passed before its
   squash. Show that they agree (atol 1e-6). On the 304 requests of
   §19.52 (each of nineteen sentences asked for each of fifteen
   words and for "words"), count the requests that return exactly
   the asked words that are present, beside the plain gradient and
   beside the rows' entries alone: the table of §19.60 on this bank.
4. Parts present by degree (§19.60, "Not measured"). Set the open
   run's activities between 0 and 1: a whole word at .3; half of a
   word's parts at .5; a word whose parts fall from 1 to .2 along
   it; the same word twice, at 1 and at .4. For the three ways of
   step 3, give what is returned and at what value. Where the map by
   weights and the rows' entries alone differ, say which a reader
   would call right, and why.
5. The where match of rule 47. For this step a concept is given
   weight on the where by the test; the model makes none. Try
   (a) the form §21 recommends; (b) the product of the two capped
   shares; (c) any other you would propose. Try them with the two
   ways a concept can read the where: (i) the Sigma's weights on the
   where coordinates of the combined vector; (ii) entries of a row
   of the Pi on the where wires and on their complements. For each
   pair: on how many of the sixteen sentences does "words" with
   that concept return the word at the place; does "words" alone
   return every word, with none lowered; how do the other tokens
   fall as the weights grow (the table of §20.5); is the asked word
   cut by its own place.
6. Whether location is learned with no location concept made (rule
   47; §19.63, point 4). The request is a learned map from what has
   been read (the places and times of the words taken so far, which
   is what AttentionContext.context gives of the understood prefix)
   to a mask over the concepts, starting at "words" alone. The
   weights that read the where start at zero and are learned. A
   form that was taken is lowered in the bank's activity over its
   extent. The loss asks that the next word in order be the most
   active of the returned set, and its gradient passes through the
   salience map, the where match and the way up. Teach minimal
   sentences first: one word, then two, then the lesson's. Report
   what was taught, for how many steps, with which optimizer and
   rate; how many of the sixteen sentences are then read wholly in
   order, as a curve over the steps; which weights came to read the
   where; and the same with the gradient on the request alone. Run
   it from five different starts and report each one. Fix no seed
   so that it comes out. It is a learning result: report it as it
   is.
6b. Added on 2026-11-19, from §19.65 and rule 49. Alec: "Reading in
   order will create location concepts as long as we are teaching
   location. So we might teach the reconstruction over bytes as built
   through a conjunction of the what and where pathways; recreated
   percepts have recreated locations that place them in the out
   buffer, so the error is over both pathways." Build that
   reconstruction on this bank: on the way back a concept gives each
   percept it recreates a band of where wires, through the weights by
   which it reads the where; the band is matched against the code of
   every place (first the product over the wires and their
   complements, as §20.5's place row; then any other form you would
   propose); the match is the percept's weight at that place of the
   out buffer; what is put back at a place is the sum over the
   percepts of what each is, times its weight there; the error is
   against the sentence's bytes at their places. Then run step 6
   three ways: taught by the reading loss alone, by this
   reconstruction alone, and by both. Report for each what step 6
   asks, and which concepts came to have weight on the where. The where
   parameter is fixed in all of them (Alec, §19.66); the run with it
   given to the optimizer, asked for here at first, is not needed.
7. If step 6 does not come to read in order, build the fallback of
   §19.48 beside it (a network gives the .where outright; from that
   extent the largest word is taken, or failing a word the largest
   concept), taught the same way, and report it as in step 6.
8. On this bank: "words" presses against the space (certificate
   34), and a word among garbage letters is returned alone
   (certificate 35).

Tests of the claims of steps 3, 5 and 8 go in
test/test_item17_looks_prototype.py or a file beside it. A learning
result is printed by the report and not asserted. Run those test
files, and name them.

Return as §23.2: the files; each result as a table; what surprised
you; which form of the where match and which way of reading the
where you would build in round 6; and what you would change in rules
34 to 39 and 47 of §21.
~~~

### 22.3 Round 3: order zero, and rows by nomination

~~~
Round 3 of item 1.7. Rules 6, 7, 8, 8b, 19, 20 and 21 of §21.
Certificates 5, 11, 12 and 23 (a) to (c). It starts from the source
that round 1 returned. Read §18.24, §19.10 and §19.17 to §19.20.

1. Before any edit, take list 4 of §19.4 and the rows of §19.21's
   table that name order zero again on your starting source. Return
   any reader that is new, gone or different, with what you build
   for it.
2. A basic percept is a row of PiLayer3 over its parts and wholes
   (rule 6), in place of the least of BasicPercepts.read. For order
   zero store.conjunctive and the key conceptualPi go. Port the
   configurations that set the key (true in
   data/MM_corpus_lessons.xml, MM_corpus_orders.xml, XOR_exact.xml,
   MM_xor_fixture.xml, MM_xor.xml, XOR_symbols.xml and
   BasicModel_corpus.xml; false in model.xml and MM_image.xml) and
   the schema.
3. Rows by nomination (rule 20). A basic percept becomes a row when
   MTM nominates it on recurrence (concept_mint_count), in place of
   the reserve at a word's first reading (BasicPercepts.reserve in
   the word loop of bin/Models.py) and of the lessons' declared
   admission (admit_training_forms of bin/Corpus.py,
   admit_native_forms of bin/ConceptLessons.py). The grammar's
   declared concepts and the frozen thought predicates are as rule
   20 has them. Two more are declared here (rule 35): "words", which
   every word's concept is made a member of when its row is
   nominated, with a Not entry on the space; and "objects", whose
   members round 4 writes. One concept for each row (rule 21): promotions of
   classes, identities and percept collections become nominations
   of rows.
4. Combination and the Sigma (rules 7, 8 and 8b): the Pi on the
   gaps over the active percepts' vectors, and SigmaLayer3 with a
   dense U. The .where and .when coordinates of a reading are not
   combined over its percepts (rule 7, from round 2): they are the
   reading's own place and time, at the degree that it is present.
   These are in place of the folds of
   ConceptualSpace._compose_order0, of
   IndexedSigmaConceptsFromPercepts and of SparseLayer._fold_weights
   at order zero. What the writers of add_concept_feature write
   today (promotion in bin/FamiliarTier.py and bin/Memory.py,
   PerceptCollections.collect, bin/ConceptLessons.py) becomes a
   column of U.
5. Storage (rule 19): the Pi of order zero as compressed rows on
   the device, with an index of columns, growing into capacity
   allocated beforehand.
6. Certificates 5, 11, 12 and 23 (a) to (c) in a new
   test/test_item17_order_zero.py. Certificate 23 is §19.20's.
7. A test that taught a word by one reading now needs
   concept_mint_count readings, or a row that is declared. Port
   each so that it states what it stated, and list each.

The orders above zero keep their folds until round 4, and the model
runs whole at the end of this round. Run: the new test file; every
test file that names what you changed; both stopping gates; the
lesson's sixteen sentences before and after, as in round 1. Return
as §23.3.
~~~

### 22.4 Round 4: symbols and the orders above

~~~
Round 4 of item 1.7. Rules 4, 9 to 14, 19 and 22 of §21, the meta
concept and the blank object of rule 20, and rule 29 above the basic
level. Certificates 4, 6, 13, 18 and 23 (d). Read §18.22, §18.23,
§18.25, §19.10 and §19.24.

1. Before any edit, take list 3 and the rest of list 4 of §19.4,
   and the rows of §19.21's table on symbols, again on your starting
   source, as in round 3.
2. Symbols are split from concepts and have no dimensions (rule 9):
   separate tensors. The key symbolTower goes, with its
   configurations ported (data/MM_20M_xor.xml, XOR_exact.xml,
   XOR_symbols.xml, MM_add_verb.xml,
   matrix/MM_20M_xor_noraise.xml). The shared row of a concept and
   its symbol is given up (the readers of list 3).
3. A symbol is two unsigned rows of PiLayer3 and brings one concept
   (rule 10). It is made at MTM's nomination, or by interpret for a
   meta concept, and not at a concept's allocation.
4. The cycle above order zero (rules 4, 11 and 12): SigmaLayer3,
   sparse, reads each symbol's pair combined into one signed input.
   The least and the greatest of SparseLayer.fold_presence go above
   order zero, and SparseLayer._fold_weights with them
   (_concept_rung_presence in bin/Spaces.py, FieldProjection.close).
5. A signed value is its two unsigned lanes on the way up and on
   the way back (§19.24): the word step's payload and the walk's own
   pair, as the row of §19.21's table says, with the readers of a
   word that is stored against traced first (1.8 specification,
   §68.63).
6. The blank object and the meta concept (rule 20): interpret makes
   both for a word with no object, in place of the pending name
   (InterpretLayer.reading, PrependedSymbols). The blank object's
   vector is zero, and no read divides by its length. Each object
   that interpret finds or makes for a word is made a member of the
   declared concept "objects" (rule 35).
7. The penalty (rules 15 and 22): §18.3's, measured first, on the
   Pi's entries and on the Sigma's entries above order zero. The
   structure of a meta concept is never thinned.
8. Certificates 4, 6, 13, 18 and 23 (d) in a new
   test/test_item17_symbols.py.

Run: the new test file; every test file that names what you
changed; both stopping gates; the lesson's sixteen sentences before
and after. Return as §23.4.
~~~

### 22.5 Round 5: the vertical spread, the order of learning, and thought's reads

~~~
Round 5 of item 1.7. Rules 23 to 28, 30 and 31 of §21. Certificates
14 to 17, 19, 20, 21, 24, 25 and 26. Read §19.11, §19.12, §19.22,
§19.23, §19.25 and §19.27.

1. The spread (rules 23 to 27): a fan up and a share down on the
   structure's own matrices, in one pass, written to
   _expected_knowing at every order (ConceptSpread.open_prediction
   writes order zero alone today). intuit runs it after it deposits
   expect's field, and it runs at the end of the read, before
   comprehension takes its leaves' objects. It adds no parameter and
   changes no evidence.
2. Not strictly ramified (rule 26): a symbol reads the concepts of
   every lower order, where a membership is admitted only between
   adjacent orders today (class_member; the row of §19.21's table).
3. The spread reaches the first layer, and the way back reads what
   it expects (rule 30).
4. The order of learning (rule 31), with §19.25's clauses and
   §19.27: an existing object is looked for before one is made.
5. Thought reads the structure (rule 28): parts and wholes of
   bin/Queries.py, and ConceptualSpace.concept_parts and
   concept_wholes, read the matrices. LTM1's part table stays the
   evidence for isPart's proofs.
6. Certificates 14 to 17 and 19 to 21 in a new
   test/test_item17_spread.py. Certificates 24, 25 and 26 are
   testpoints in test/test_testpoint.py (TestBlindSpot,
   TestCrossModalTie, TestBellSalivate), each printing the values
   that §19.23 and §19.25 name. Where 25 and 26 say "learned" they
   are certificates of learning.

Run: the new test file and the testpoints; every test file that
names what you changed; both stopping gates; the lesson's sixteen
sentences before and after. Return as §23.5.
~~~

### 22.6 Round 6: the looks, in the model

*Amended on 2026-11-19 from round 2's return and its review (§23.2, §23.2.1): steps 3, 4 and 11.*

~~~
Round 6 of item 1.7. Rules 32 to 42 and 47 to 49 of §21.
Certificates 30 to 45. Read §19.45, §19.48, §19.49, §19.53 and
§19.60 to §19.65;
round 2's return (§23.2); and Claude's review of it, which says
which form of the where match is built.

1. The where from a parameter (rule 47). The registry's encoder
   (WhereRegistry, WhereEncoding) holds one parameter with the code
   of every place, written from WhereEncoding.encode.
   WhereRegistry.encode reads its rows, and the percepts' .where
   (PartSpace._radix_code_events, _embed_radix) flows from it. It
   takes a gradient and is in no optimizer's groups. Where a stamp
   is a write into a buffer that takes no gradient today, say what
   you did.
2. The bank (rule 33). At a sentence's start, where the open run is
   made (FamiliarTier.open_sentence, SentenceField.open_field,
   _sentence_open_fields), the rows of the one lookup are held once
   on the device: each event's row, place, band, time, token and
   activity. A known form that spans several units is an event of
   its own. No materialize() and no host read of a span is on a
   look's path (stage_cs_sub, _per_word_body_step and _peer_stage_b
   of bin/Models.py call materialize() today; count them again).
3. The open run keeps the Pi's local derivatives. The salience map
   (rule 36) carries a request down by the transposes, each Sigma
   by its weights, onto each event's activity and onto its band.
   test/looks_round2.py has it both ways (salience,
   salience_by_transposes); the transposes are what is built.
4. A look (rules 34, 37, 39 and 47), as round 2 tried it
   (look and where_match of test/looks_round2.py). An event's
   activity is its share capped at its own activity. A reading's
   where is the band of the event at its start, weighted by that
   event's presence; its where share is the sum over that band of
   the gradient times the value; the where match is exp(s - max s)
   over the sentence's readings, and it multiplies a reading once,
   never inside a row's product. Then the way up again through what
   is active (cs_read_memberships with the activity in place of a
   bracket, then cs_forward_content), and the returned set.
   interpret takes the most active word concept; of forms that
   begin at one place the largest, first; among equals left at
   different places a draw, credited as a departure is
   (explorePaths, SentenceCredit). Report what the request learns of
   the what, since the cap passes it no gradient at one or above;
   and what a word present by degree comes back as.
5. attend (rules 35 and 38). OperationSelectionLayer.attend gives
   the request, a mask over all concepts on two lanes, from
   AttentionContext.context, and starts at "words" alone. The
   bracket table and its acts go (BracketTable, narrow_step,
   narrowing_view and narrowing_mask of bin/Attention.py); the done
   mark of record_interpretation becomes a lowering of the bank's
   activity over the form's extent; and Looks.run of
   bin/SentenceLooks.py is the loop of requests. The grammar's
   option at a look and interpretBias stay (interpret_at_look). Say
   what becomes of FocusLayer, UnfocusLayer and attention_operations,
   and of the grammar files that declare them.
6. The node "words" (rule 35), declared in round 3 with its Not
   entry on the space, is what the request starts at.
7. One gain field (rule 40): Expectation.gain_field takes the
   look's activity as its attention and the look's forecast as its
   expected. LookForecast.difference goes.
8. The reading lesson (rule 38): ReadingTargets and reading_lesson
   of bin/AttentionLesson.py teach the request, minimal sentences
   first. The test of reading in order gates the grammar tests
   (certificate 38 (c)), as the assessment's first rung gates its
   others (bin/AssessmentProbe.py, data/assessment/known_probe.json).
9. A literal (rule 42) and silence (rule 48): a layer whose
   rule_name is "silence", in GRAMMAR_LAYER_CLASSES and in no
   grammar file, applied where Models._stm_reduce_to_single_S finds
   that the sentence's trees did not reduce to one; _program_entries
   and _commit_sentence then write its row.
10. What goes (§21, "What goes"): Models._passback_scope_ps and
    _passback_scope_where with its two writers
    (SentenceLooks.perceive, ModelAttention.handoff); the key
    attentionFloor (bin/AttentionContext.py, bin/AttentionObjective.py,
    data/model.xml, data/model.xsd); the tests of spans on the host.
11. The reconstruction over what and where (rule 49): a recreated
    percept has a recreated band, which places it in the out buffer;
    the error is over both. Read where the source places what it
    puts back today (SentenceField.native_target, reverse of
    bin/FieldReverse.py, Models.reverseReconstruct). The match of a
    recreated band against the places is a softmax over the places
    of the out buffer (reconstruct and place_weights of
    test/looks_round2.py). It is taught beside the reading lesson.
12. Certificates 30 to 45 in a new test/test_item17_looks.py. A
    test that reads a model supplies its reading and does not leave
    it to the chooser's draw (1.8 specification, §66.9). The
    prototype's three files go when the model's own tests state what
    they stated; say which.

Run: the new test file; test/test_loop_of_looks.py and every test
file that names Attention, ModelAttention, SentenceLooks,
AttentionLesson, AttentionContext, LookForecast or WhereRegistry,
each in a fresh process; both stopping gates; the lesson's sixteen
sentences before and after, with the host reads of a sentence.
Return as §23.6.
~~~

### 22.7 Round 7: the readers, the measurements and the sweep

~~~
Round 7 of item 1.7. Rules 16 and 17 of §21. Certificates 8, 9, 10
and 44. It is the round that may be pushed; the push is Alec's word.

1. Take lists 1 and 2 of §19.4, and what is left of the tables of
   §19.21 and of §21, again on your starting source. Likeness of
   codes pairs nothing (rule 16): each reader of list 1 as its row
   says; the candidates of MediumTermMemory.match_candidate,
   FamiliarTier.witnessed_match and FamiliarTier.observe_field by
   parthood both ways (ConceptualMatch.match), with the bar set by
   a trial that is reported. The way back (rule 17): each reader of
   list 2 as its row says. Certificate 8.
2. Search the source for what the item's rules replaced and is
   still there: a key that no reader reads, a function that no
   caller calls, a second path. Remove it, and list it.
3. On the finished source, once: both stopping gates; the lesson's
   sixteen sentences with their readings, each order's occupancy,
   the Pi's kept edges for each row and the Not entries at each
   order, beside item 1.8's last return (1.8 specification, §66.3
   and §66.14); the seconds of the sixteen sentences and the host
   reads of a sentence (certificates 10 and 44).
4. Then the item's one default sweep, on this finished source only:
   env -u RUN_TRAINED -u RUN_SLOW make test (1.8 specification,
   §72). Nothing under RUN_SLOW. If make plans to rebuild .venv
   because requirements.txt is newer than .venv/.installed-bin,
   check that nothing is missing (pip install --dry-run -r
   requirements.txt) and touch the stamp. The verbal proof and the
   assessment's tables are read from the sweep's report. A failure
   is diagnosed to its producer, repaired if it is this item's, and
   the sweep is run again; the return gives every run.
5. Return as §23.7: the lists of step 1; what step 2 removed; the
   measurements of step 3 beside item 1.8's; the sweep; and the
   source snapshot (test/bounded_tests.source_snapshot).
~~~

## 23. Item 1.7: the returns of the rounds

Each round's return is a section here, §23.1 to §23.7, written by its
implementer, with Claude's review after it.

### 23.2 Round 2: the looks, tried in tests only (Claude on Opus, standing in for Codex, 2026-11-19)

No file of `bin/` and no existing test was changed; the prototype's three
files were left as they were. Nothing is wired into the model. Every
form below that the rules leave open is a choice of this trial, built in
its test files only, and so provisional throughout.

**The files.**

- `test/looks_round2.py`. `part_map` and `Lexicon` (rule 43, built in
  the test: characters with the space, a run added only where two forms
  have the same parts, carried by every form that holds it; on this
  vocabulary one part is added, `ac`, for `cat` and `act`). `Bank.read`
  (rule 33: for each stem unit its form's row where known and its parts'
  rows at their places; `ice cream` a token of its own at its start).
  `Trial.up` (rule 2's Pi, rule 7's combination with the four where
  coordinates, rule 3's Sigma with one concept for each row, "words"
  with a Not entry on the space; `Trial.theta`, the where parameter,
  written from `WhereEncoding` at capacity 4096, in no optimizer).
  `salience` (framework backward) and `salience_by_transposes`
  (rule 36). `look`, `where_match`, `interpret`, `returned`. `Reader`,
  `context`, `read_in_order`, `teach_order` (step 6). `reconstruct`,
  `place_weights`, `shift_band`, `rebuilt_letters` (step 6b, rule 49).
- `test/looks_round2_report.py`, which prints every table below.
- `test/test_item17_looks_round2.py`, 21 tests of the claims of steps 1,
  3, 5 and 8.

**My choices, each needed to run it** (commit `a0f9bef801` unless said).

1. A row's what is its own coordinate at value 1 − 1/e, so rule 7 gives
   1 − exp(−b): graded, with a derivative everywhere (at value 1 it is
   0^b, which has neither).
2. A concept reads its own row at G = 3; "words" reads each word concept
   at G and has a Not entry of 2G on the space's concept, so that a part
   holding a space has a net negative share.
3. A form's Pi row has weight 1 on its own row and on each part of its
   code (a set, so 1, not a count).
4. The where of a reading. Rule 7 as written combines the bands of every
   event of a reading ("combined"). Two others were tried: the band of
   the event at the reading's start alone, by its Pi value ("head"); and
   that band whatever is present ("start").
5. Places are sentence-local from 0, at the prototype's capacity 4096
   (short period 256, origin shift 128).
6. `interpret`: of equals that begin at one place the largest is kept;
   equals left at different places are a tie, and a tie is never counted
   as reading the word.
7. Step 6's context: four slots of the taken words, each the code of its
   start, occupied, and its age; and a bias. The request is linear in
   it, over "words" and every row's concept, starting at "words" alone.
   The done mark sets a taken token's activity to 0. Each look's map is
   taken from a run on the current activity. Training is teacher-forced.
8. Step 6b (commit `0c6f4dfe5f`): a token's own word concept recreates
   its characters in order (rule 45). The recreated band is the place
   the concept's where weights read best, ½ + U/2, moved on by each
   character's offset by rotating each rung. The error is squared, over
   one coordinate per row, at every place of the sentence.

**Certificate 31 (a): transposes against the backward** (atol 1e-6
asserted).

| Where of a reading | Sigma passed by | Comparisons | Largest difference |
| --- | --- | ---: | ---: |
| combined | slope | 160 | 3.6e-14 |
| combined | weights | 160 | 2.3e-12 |
| head | slope / weights | 160 / 160 | 3.6e-14 / 2.0e-12 |
| start | slope / weights | 160 / 160 | 3.6e-14 / 7.1e-14 |

**Step 3: the 304 requests** (19 sentences, 15 words and "words";
returned = word concepts at 0.01 or more).

| Carried down by | Exactly the asked words present | Misses one present | Returns one not asked |
| --- | ---: | ---: | ---: |
| the plain gradient (Sigma by its slope) | 229 | 75 | 0 |
| the gradient, each Sigma before its squash | 304 | 0 | 0 |
| the rows' entries alone | 304 | 0 | 0 |

A word that is absent returns nothing; `Fe li x` and `xileF` never return
`Felix`, since a form's row requires its own row.

**Step 4: parts present by degree** (`Here is Felix.`; the value of
`Felix`; "times" is rule 34, own activity times the share capped at one;
"min" is the share capped at the event's own activity).

| Case | Open run | Asked `Felix`: by weights, times | entries, times | entries, min | Asked "words": by weights | entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the whole word at .3 | .002 | 0 | .001 | .002 | 0 | .002 |
| three of five parts at .5 | .339 | .000 | .339 | .339 | .326 | .339 |
| parts falling 1 to .2 | .113 | 0 | .068 | .113 | 0 | .113 |
| twice, at 1 and .4 (the second) | .012 | 0 | .012 | .012 | 0 | .012 |

The plain gradient returns nothing in any case. Every way that returns
anything takes the same word (`Felix`, and in "twice" the copy at 1). They
differ in value. Under the gradient by weights a part's share is
G² e^{−b} b, the presence of the row it is in, so a row below about .11
gets a share under one and is cut a second time on the way up. A reader
would call the rows' entries right on value: the look returns the word
at the degree the open run found it, and it takes away nothing that the
request asked for. The gradient's ranking is input-sensitive as Alec
wanted, but here it shows only as cutting weak words. Rule 34 also
counts the activity twice: the share already carries it. The "min" rule
fixes that and gives the open run's values exactly by entries.

**Step 5: the where match** ("words" with a concept given weight on the
where by the test, at each word's start in turn, on the 16 sentences;
sentences wholly right / places right of 48; k is the magnitude).

| Where of a reading | Way | (a) §21's form | (b) capped share | (c) forward reading |
| --- | --- | --- | --- | --- |
| combined | (i) Sigma weights | 0 / 0 at every k | 0 / 0 | 0 / 16 at every k |
| combined, head, start | (ii) Pi row | 0 / 16 at k 12.5, 0 / 0 at 800 | 0 / 0 | 16 / 48 at every k |
| head | (i) | 16 / 48 at k 1 to 100; 0 / 32 at 1,000 and 10,000 | 0 / 28 at k 1, else 0 | as (a) |
| start | (i) | 16 / 48 at every k (1 to 10,000) | 0 / 28 at k 1, else 0 | 16 / 48 at every k |

- "Words" alone, nothing of where asked: every word returned at its open
  value, none lowered, in 16 of 16 for (a) and (c), and 0 of 16 for (b),
  which cuts everything when nothing of where is asked.
- *(a) with (ii) cannot work.* A peaked Pi row has a gradient of zero at
  its own place, so the grad·value share is zero there. This is a test:
  `test_the_where_share_of_a_peaked_pi_row_is_nothing_at_its_own_place`.
- *Rule 7 as written blurs the where.* It combines the bands of every
  event of a reading, five or six for a word. The coordinates then
  depend on how many events there are more than on the place: the
  long-rung cosine is .998 at every place and saturates, and the short
  cosine adds up over the events. The where of a reading has to be its
  start's band.
- *Head, at large k, the asked word is cut by its own place, but through
  the what.* "Head" weights the band by the head row's presence, so the
  where concept's gradient also lands on the activities, with either
  sign. At k = 1,000 it lands negatively on the asked word's events and
  their what share clamps to 0. On "This is Felix." asking `Felix`'s
  place, with head, (i), (a) and applied once: This/is/Felix are
  .947/.955/.956 at k 1, .365/.835/.956 at k 100, and 0/0/0 at k 1,000.
  "Start" has no such path and holds at every k.
- *Applied inside the products* (each event at its own place), (a) cuts
  the asked word at every k (`is` taken at k 1 for `Felix`). Applied
  once to the reading, it does not.
- (c) with (ii) dims the others to e^{−G} = .048: This/is .115/.773 at
  k 12.5 and .048/.048 at 800, with `Felix` at .956 throughout.

**Step 6: reading in order, no location concept made.** Curriculum: one-
word sentences (10 steps), two-word (10), the lesson (60); Adam, rate
0.05, full batch; the request starts at "words" alone plus noise of 0.01
on its other entries, from a seed drawn from the system and printed; the
where weights start at zero. "Held out": six sentences of known forms in
orders the lesson never shows (`is This Felix?`, `Ciao is that cat.`).
"Match off": the learned reader with the where match set to one.

| Where of a reading | Taught | Starts reading all 16 in order | When | Held out | Match off | Weights that came to read the where |
| --- | --- | ---: | --- | ---: | ---: | --- |
| combined | request and where weights | 5 of 5 | lesson step 10 to 20 | 4, 4, 4, 4, 4 of 6 | 8, 10, 16, 16, 16 | word concepts (`there`, `Felix`, `Ciao`), norm 1.0 to 1.7 |
| head | request and where weights | 5 of 5 | two-word step 5 | 6 of 6 each | 0 each | character concepts the request asks (`x`, `F`, `T`...), 1.7; word concepts 0.82 |
| start | request and where weights | 0 of 5 (16 at two-word step 5, then lost; ends 12, 12, 12, 12, 12) | | 2 to 4 of 6 | 0 | `i`, `T`, `There`, 2.3 to 3.0 |
| each | the request alone | 0 of 15 | | 0 | | none |

- *It is learned with no location concept made,* in 5 of 5 starts with
  combined and 5 of 5 with head. Only under head is the where match
  doing the work: it reads held-out orders 6 of 6, and with the match
  off it reads nothing. Under combined, the word concepts' where
  weights act in the forward pass and the order is tied to content: 4
  of 6 held out.
- *Start fails for a reason of its own.* A concept's where weights light
  it at a token where its form is absent: `is` at the `?` of "Who is
  here?". The where must be weighted by presence.
- *The request alone learns nothing,* and its gradient is exactly 0
  (checked at the start: loss 28.67 = 16 ln 6, gradient norm 0). "Words"
  at 1 gives every word event a share of about 3.3, the cap at one holds
  it there, and the cap passes no gradient.
- Each run takes 2.0 to 2.2 s on the host.

**Step 7 was not done:** step 6 reads in order.

**Step 6b: the reconstruction over both pathways** (rule 49). As step 6,
five starts each; the where parameter is fixed (§19.66). "Rebuilt" is the
lesson's letters whose own character has the most weight at their place,
of 164.

| Where of a reading | Taught by | Match | In order (starts) | Held out | Rebuilt | Where weights on |
| --- | --- | --- | --- | ---: | ---: | --- |
| head | reading | | 16 (5 of 5) | 6 | 18 to 22 | characters 1.4, words 0.8 |
| head | reconstruction | product (§20.5, k 200) | 0 (0 of 5) | 0 | 10 | nothing: 0.00 |
| head | reconstruction | softmax (mine) | 16 (5 of 5), from one-word step 10 | 4 | 78 | word concepts only (`That`, `Here`, `There`), 1.36 |
| head | both | product | 16 (5 of 5) | 6 | 18 to 20 | characters 1.45, words 0.83 |
| head | both | softmax | 16 (5 of 5), from one-word step 10 | 6 | 77 | characters 2.45, words 1.56 |
| combined | reading | | 16 (5 of 5) | 4 | 13 to 27 | words 1.15 |
| combined | reconstruction | product | 0 (0 of 5) | 0 | 10 | nothing |
| combined | reconstruction | softmax | 0 (0 of 5) | 0 | 78 | words 1.35 |
| combined | both | product | 16 (5 of 5) | 2 to 4 | 17 to 19 | words 1.05 |
| combined | both | softmax | 16 (4 of 5) | 3 to 4 | 77 to 80 | words 1.4 |

- *The product match never starts.* A recreated band at rest (½ on
  every wire) is far from every place's code: a long-rung cosine of .998
  and a short one near 0. At k 200, and at 12.5 when tried, the match is
  zero everywhere and passes no gradient. The softmax over the buffer's
  places learns at once.
- *The reconstruction alone teaches location,* and under head it reads
  in order with no reading loss (5 of 5). What it makes are word
  concepts that hold the place their word usually has: first words
  early. So 4 of 6 held out read, against 6 with the reading loss.
  Rebuilt stays at 78 of 164: a concept has one where for all its
  occurrences, so `is` at places 4, 5 and 6 cannot all be put back.
- *Best: head, both teachers, softmax.* It reads 16 of 16 from the first
  stage, holds out 6 of 6, and rebuilds 77.
- *A note, kept though §19.66 dropped it.* The best run, five starts
  with the where parameter in the optimizer: 16 of 16 in four starts
  and 12 in one; held out 2 of 6 in each; rebuilt 34 to 53. Its values
  moved by up to 0.69 to 1.0 on places 0 to 19, so the code no longer
  means the same place. Without it in the optimizer: 16 of 16, 6 of 6,
  77, and a change of 0.

**Step 8.** With "words", `ice cream` is at 0.000 and `ice` and `cream`
are at .956, and the space event's what share is 0. With no "words", a
place at 8 (Pi row, (c)) takes `ice cream`, the largest of the two forms
there; "words" with the same place takes `ice`. On the lesson's 16 no
part selected by "words" holds a space. `qzv Felix xwpt.` returns
`{(1, Felix)}` alone, and `Felix` is taken. `qzFelixvb.`, the word inside
a run of garbage letters, returns nothing: the stem's units hold no form
inside a unit until the lookup finds parts at every place (rule 44). The
declared "objects" was not added.

**Tests run** (BASICMODEL_DEVICE=cpu, RUN_SLOW and RUN_TRAINED unset):

- `test/test_item17_looks_round2.py` with `test/test_item17_looks_prototype.py`:
  30 passed, 6.1 s.
- Both stopping gates (`test_mm_xor.py::TestMMXorConvergence::test_convergence`,
  `test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class`):
  2 passed, 3.8 s, on the tree with round 1's uncommitted edits of
  `bin/`.
- `test/looks_round2_report.py` printed the tables, in about 4 minutes.

No test was ported or retired.

**Commits.** `a0f9bef801` (the trial and its report), `cfafe38aef` (its
tests), `0c6f4dfe5f` (step 6b), and this return.

**What surprised me.**

1. Rule 7's combination makes the where of a reading count its percepts.
   No form of the match can read a place through it.
2. The grad·value share of a peaked row is zero at its own peak.
3. The cap at one leaves a request that asks for "words" at full no
   gradient at all.
4. The reconstruction alone taught reading in order. It did so through
   where the words usually are, which is content and not position.
5. Under head, the location came to be carried by character concepts
   that the request asks for at about ±2. Those are concepts of
   location over the network, as §19.63 expected, though not the ones
   anyone would name.

**Recommended for round 6.** Build the where match as (a), exp(s − max s)
on the salience map's share over the head band, applied once to a
reading. The where should be read by way (i), the concepts' Sigma
weights on the where coordinates. Those coordinates should be the band
of the event at the reading's start, weighted by its presence ("head"),
and not rule 7's combination over every event.

Why:

- It is Alec's pathway: the map lands on wires and the concepts' weights
  read them.
- It learned in order in 5 of 5 starts, held out 6 of 6, and its match
  is what does the work.
- Combined fails step 5. Start lights absent concepts. (a) with (ii) has
  no gradient at the place. (b) cuts everything.

Its limit is the leak onto the what at magnitudes near 1,000. Learned
magnitudes stayed under 3. Teach both losses, with the softmax match for
rule 49.

**What I would change in §21.**

- *Rule 34:* the look's activity is the share capped at the event's own
  activity, min(own, share), in place of own times the capped share. The
  share already carries the activity.
- *Rule 36:* keep the gradient by weights, as decided. Note that it cuts
  a row present below about 1/G² of its gain. Also note that the cap of
  rule 34 gives no gradient to a request already at or above one: at
  the start the request learns only through the where.
- *Rule 37:* of equals, the largest at each place first, then a tie if
  places differ, as built here.
- *Rule 38:* the done mark is enough for order once the where is learned.
  No step needs teaching.
- *Rule 39:* the where match is applied once to a reading, never inside
  the products. Inside, it cut the asked word in every form tried.
- *Rule 47:* state that the where coordinates of a reading are its start
  event's band, weighted by its presence, and not rule 7's combination.
  State (a) as the match.
- *Rule 49:* the match of a recreated band against the places as a
  softmax over the buffer's places. The product of §20.5 passes no
  gradient from rest.
- *Rule 7:* say that .where and .when are not combined across a
  reading's percepts.

### 23.2.1 Review of round 2 (Claude, 2026-11-19)

**Accepted as a trial.** It changed no file of `bin/` and none of the
prototype's three files (the difference of its four commits, read). Its
tests pass here with the prototype's: 30 passed in 5.3 s. The code does
what the return says: the where parameter is written from
`WhereEncoding` and given to no optimizer, each start's seed is drawn
from the system and printed, and no seed is fixed for a result.

**Run again by Claude, ten fresh starts** (`looks_round2_report.run`):

| Where of a reading | Taught by | In order, of 16 | Held out, of 6 | Rebuilt, of 164 |
| --- | --- | --- | --- | --- |
| head | reading | 16, 16 | 6, 6 | 18, 20 |
| head | reading and reconstruction, softmax | 16, 16 | 6, 6 | 77, 77 |
| head | reconstruction alone, softmax | 16, 16 | 4, 4 | 78, 78 |
| combined | reading | 16, 16 | 4, 4 | 13, 21 |
| head | the request alone | 0, 0 | 0, 0 | 10, 10 |

They are the return's numbers.

**What it settles, for round 6.**

1. *Reading in order is learned with no location concept made,* on the
   lesson's sentences, from every start tried. The fallback of §19.48
   is not built.
2. *The where match is §21's form:* the where share referred to the
   most salient reading, through the exponential, applied once to a
   reading. The capped share cuts everything when nothing of where is
   asked. A row of a Pi on the where wires has no share at its own
   place, so the where is read by the concepts' weights, as Alec has it.
3. *A reading has one where, its own.* With the bands of all of a
   reading's percepts combined as rule 7 combines vectors, no form of
   the match reads a place (0 of 48): the coordinates then say how
   many percepts a reading has. With the band of the reading's start,
   weighted by its presence, the place is read in 16 of 16.
4. *The reconstruction teaches location,* as Alec said it would
   (§19.65). Alone it gives concepts the place their word usually has,
   which is content (4 of 6 held out). With the reading lesson it reads
   6 of 6 and puts back 77 of 164 letters. The match of a recreated
   band against the places has to be a softmax over the places: the
   product of §20.5 gives no gradient from rest.
5. *The where parameter fixed* is right in the numbers as well: given
   to the optimizer, its values moved by up to 1.0, and held-out
   sentences fell from 6 of 6 to 2.

**Three things it found that are not what the rules assumed.**

1. *The salience map by weights cuts weak words.* A whole word present
   at .3 comes back at 0, and so does a word whose parts fall from 1 to
   .2, where the rows' entries alone return what the open run found.
   The share goes with the presence of the row, which is a product, so
   it falls fast. Alec decided the map by weights (§19.64, point 1) and
   it stays; this is how it behaves where parts are present by degree,
   which §19.60 left to be tried.
2. *The cap at one gives the request no gradient on what it asks.*
   "Words" asked at one gives every word a share above one, and the cap
   passes nothing back. In the trial the request learned only through
   the where. Round 6 reports what the request learns of the what.
3. *Location came to be carried by character concepts* that the request
   asks for, not by concepts anyone would name. They are concepts of
   location over the network, as §19.63 expected.

**Limits of the trial** (*Claude's*). Sixteen short sentences; a
context of four slots made for it; teaching with the right word given
at each look; one concept has one where for all its occurrences, so
half the letters are not put back. At magnitudes near 1,000 the where
leaks onto the what and cuts the asked word; what was learned stayed
under 3.

**Certificate 35, as it stands.** A word among garbage letters is
returned alone where the garbage is separate runs of letters (`qzv
Felix xwpt.`). A word inside one run of garbage (`qzFelixvb.`) returns
nothing, since the stem's units hold no form inside a unit; that needs
the parts found at every place, which is the lookup of rule 44 in item
1.5. For Alec.

**Taken into §21** (each *mine*, his to change): rules 34, 37, 39, 47
and 49 as the return recommends, and rule 7 for `.where` and `.when`
alone. Rule 7 is a rule he confirmed, so that one is put to him. Round
6's text is amended to match.
