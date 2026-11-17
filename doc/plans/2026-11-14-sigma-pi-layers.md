# SigmaLayer and PiLayer: a sum, a product, and no chart that only bounds (Claude, 2026-11-14)

A specification for rewriting `SigmaLayer` and `PiLayer` of
`bin/Layers.py`. It was round D of item 1.8 (§18). Since 2026-11-16 it
is item 1.7 (Alec: "I think we should position D as item 1.7"), built
after item 1.8 is finished and pushed; it keeps the name round D here.
It has no text for Codex yet: that is written on the source item 1.8
leaves. A rule marked *default* is Claude's choice where Alec has not
spoken. Not yet written here: the discussion of 2026-11-16 after §18.23
(flattened percept vectors read by one template per concept, one Sigma
and Pi form at every order, an event axis for symbols, binding a letter
to its place).

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

**The two classes, updated where they stand.**

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
