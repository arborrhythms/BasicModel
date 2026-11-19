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
    returned token is.

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

### 19.41 Placed in item 1.6 (Alec, 2026-11-18)

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
