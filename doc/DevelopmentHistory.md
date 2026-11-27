# Development history

Sections moved here from the main documents because they describe
mechanisms, proposals or measurements the current contract has superseded.
Each is as it stood, with the labels of the development process removed
(2026-11-12), under a heading naming its source document, with the
lines it occupied there on the day it moved. The main documents keep a
pointer at each site. Nothing here governs; the governing records are the
current documents. The store names changed on 2026-10-21: LTM0 is the
occurrence store (LTM1 in older text here), LTM1 its relation stratum (older
LTM2), and the old LTM3 name is retired.

# Architecture.md

*From `Architecture.md` lines 211-240, moved 2026-10-30.*

> **Status (2026-05-29 update):** further architectural pivots landed
> on top of the 2026-05-27 substrate refactor:
>
> - **PerceptStore $\to$ RadixLayer.** The radix-trie input encoder is
>   now a first-class `Layer` subclass in `bin/Layers.py`;
>   `PartSpace.reverse` invokes `RadixLayer.reverse` for the
>   structural decode.
> - **MetaLayer $\to$ SymbolizeLayer.** The binary GrammarLayer that
>   promotes a freshly-seen percept to a symbolic prototype is now
>   `SymbolizeLayer`; no semantic change.
> - **Auto-META moves PS $\to$ CS.** The cross-codebook bind (META entry:
>   PS chunk-id $\leftrightarrow$ SS prototype-id) fires from
>   `ConceptualSpace._maybe_autobind_meta` at stage 0; PS no longer
>   holds a back-ref to SS.
> - **Clean-stack STM.** `ConceptualSpace.forward` bypasses
>   `sigma_in` / `sigma_cs` on forward — `folded = primary` at stage
>   0, `folded = sym` at k > 0. The Stage-10 additive composition is
>   retired; per-stage space-role attribution is trivially invertible.
> - **`basis=` kwarg for grammar reverses.** `UnionLayer.reverse(parent,
>   basis=None)` / `IntersectionLayer.reverse(parent, basis=None)`
>   accept a Codebook / Basis object (typically
>   `WholeSpace.subspace.what`) instead of a raw `W` tensor;
>   `bin/Language.py::unreduce()` dispatches accordingly.
> - **LSE soft-max kernels.** `Ops._disjunction_kernel` /
>   `Ops._conjunction_kernel` default to LogSumExp smooth variants
>   when `monotonic=False`; the hard branch is retained for
>   monotonic-mode and exact idempotency tests.
> - **LBG-style WS property splitting.** Gray (1990) EMA + per-row variance
>   tracking can split a dynamic property along its top-variance
>   eigendirection. This grows properties, never concepts or symbols.

*From `Architecture.md` lines 242-256, moved 2026-10-30.*

> **Historical status (2026-05-27; ownership superseded 2026-07-20):** the
> **substrate refactor** had landed end-to-end.
> PS is a single-arg input processor (synthesis front end + sigma fold). CS is a STM container +
> grammatical CPU (no atomic forward fold; sigma_percept retired). The paired
> word-codebook ownership described then no longer belongs to WholeSpace. The
> CKY `Chart` and STM shift-reduce parsers retire entirely; `LanguageLayer`
> (signal router) is the canonical parser. `LiftLayer` / `LowerLayer` are
> binary `GrammarLayer` subclasses with internal Sigma / Pi (no longer
> borrowing substrate folds). `GrammarLayer` gains an optional butterfly
> cascade mode for cross-position mixing — closes the XOR convergence
> target. Two operating modes selected by `<serial>`: **SERIAL/GRAMMATICAL** (per-word PS with
> grammar dispatch over STM) and **PARALLEL** (T = `<subsymbolicOrder>`
> iterations of PS over CS). The `<parserBackend>`, `<routerKind>`,
> `<chartTau>`, `<chartTopK>`, `<chartNoiseEps>` XML knobs are retired;
> `<symbolicOrder>` is now the symbolic / relational loop budget.

*From `Architecture.md` lines 340-357, moved 2026-10-30.*

> **Historical note (forward composition superseded 2026-07-10).** The v3
> iterated-symbolic-loop design (landed 2026-07-03)
> ran the store as an ITERATED WAVE with an additive source term every
> step, $a^{i+1} = \tanh(W [a^i \mid 1] + s)$, $i = 0..K{-}1$, over a row
> space split 50/50 into a SNAP block (rows $[0, n_{\text{snap}})$,
> order-0 concepts, no in-edges) and a RELATION POOL (the rest, minted
> relations, first-come with a loud overflow warning). A fresh probe on
> `MM_sparse_concept` found the wave DARK end-to-end -- sign-then-clamp
> annihilation at the order-0 rectifier, scale-blindness between the
> settled field and the codebook, and a capacity gap where `<nVectors>`
> never reached the per-stage store (the dual-towers rev-2 design's
> "Why"). Rev 2's correction: the concept base is the 8-tile corpus-callosum
> frame, not the codebook inventory, so attention is a top-K taper over
> that frame rather than a settling recurrence -- replacing the wave with
> the feedforward pyramid above closes the darkness (no trivial fixed
> point left to go dark). The `(n_snap, n_pool)` split above is the only
> surviving trace of the v3 design, kept as the `_order_caps` off-path
> fallback.

*From `Architecture.md` lines 379-419, moved 2026-10-30.*

The two-truths clause closing
uses the native concept allocator for word and object addresses. An ended
clause's identity is its occurrences in LTM, tied by the references in their
slots; it takes no concept inventory row. Each LTM row has `row_ids`, three `refs`, an `order` stamp, independent
`c_plus`/`c_minus` identification evidence, a separate scalar source `trust`,
and the ended field's four-coordinate `.where` and `.when`. Trust concerns
the event's veridicality; changing it never replaces the evidence pair.
An abstract row is decoded through the sigma inverses down its stamped
order. An idea occupies
one point slot containing the actual end state. Its two pre-fusion operand
references are a cache; no factored target or derivation persists. A
relation occupies three slots and has kind `part`, `implies` or `operator`;
there is no catch-all relation. Referenced relations have null point slots
and are read by row identity, including implication over two relations.

The September 29 amendment adds a fourth relation kind, `REL_DEF`:
`word DEF object`. Its operands name native identities in `refs[0]` and
`refs[2]`; their vector slots are null, and the middle slot has the fixed
DEF atom. Learning either concept's code cannot redirect the definition.
Word and object operands belong to the grammar registry's ConceptualSpace,
including bodies with independent stage dictionaries; another stage's allocator
cannot supply their identities. The DEF utterance and each ended clause have
stable LTM source-address keys. A grammar predicate has one identity per operation,
tied by the relation slots in which it occurs, without an inventory row. A
referenced phrase likewise keeps its point in its LTM row. An identity-to-row
cache locates these occurrences. A full inventory refuses optional
symbolization atomically and the reading continues.
Only `interpret` writes definitions; only the closing writes what a sentence
asserts. A definition is keyed by `(DEF, word, object)`, has an unlocated zero
`.when` band, remains in recency, and may be forgotten. Re-reading refreshes
its timestamp without changing its address.

The taxonomy is the native concept hierarchy, read by sigma union. A part
row between object concepts places the parent one order above its child;
word identities are never taxonomic operands. One index derived from DEF
rows supplies form/unit → word, word → objects, and object → words, without
lookup scans. It rebuilds on load and store compaction. The word/object META,
its fold and its part edge are retired in every reading; old checkpoint
bindings migrate to DEF rows. WholeSpace owns only properties. Evidence and
source trust stay on the LTM row. Clause admission and witnessed context
discovery write the hierarchy; testimony does not pass the discovery gate.

*From `Architecture.md` lines 437-447, moved 2026-10-30.*

**Groundedness and cycles -- REMOVED (2026-07-10).** `cs_groundedness_probe`
does not exist in the current codebase (zero grep hits). It was the KRIPKE
grounded/ungrounded reading of the v3 iterated wave, in two runs (source-driven,
then source-released) -- a diagnostic over settling dynamics that the
feedforward pyramid does not have. The design pass that replaced the wave
accepted the loss outright ("the wave was dark anyway," per the rev-2
design's "Theory"). The posture on cycles it used to report stands as a design
statement even without the probe: loops are a documented FACT -- of
un-ramsified taxonomies and of human minds -- and nothing in the current
codebase observes or damps them at runtime. Solutions are invited -- for
both human and machine minds.

*From `Architecture.md` lines 854-861, moved 2026-10-30.*

The legacy `SubwholeSpace` and `SyntacticSpace` classes have been
retired. The subsymbolic role is filled by `PartSpace`; syntax /
grammar dispatch lives on `SymbolSubSpace.languageLayer` (the signal router,
which subsumes the retired `Chart`). The `MereologicalTree` sidecar that
backed `part` / `equals` / `query` is also retired --- those operations are
pure-geometric clipped-cosine projections over WholeSpace codebook
activations.


*From `Architecture.md` lines 1492-1514, moved 2026-10-30.*

**Historical (Alec, 2026-09-25; row `.where` superseded above): every LTM
row carried a `.where` and a `.when`**, the field's pair, written once by
the closing. **`.where`
uses one address space over input and stored percepts**: input positions
come first, then PartSpace and WholeSpace rows, then one address per symbol
pole. A word's percept and symbol occurrences carry its input byte start,
not its stored row's location. Thought-produced symbols use their symbol-row
address. The same row can therefore appear at several input locations
without reserving extra symbol slots. Codebook identity stays the row index
per Spaces' Codebook Uniqueness Contract.
Stable slices use fixed physical capacity for every perceptual codebook,
with logical admission into existing rows. Exhaustion names
`nVectors`; it never replaces a Parameter or migrates optimizer state.
The `.where` band itself carries the registry address.
One model-owned two-rung ladder is shared by input, part, whole and symbol
spaces. Its long period covers the entire fixed registry; its short period
resolves adjacent locations. Captured programs retain these bands, and an
integer location is decoded from them when needed. There is no second saved
integer address and no separately configured 8192-byte period. `.when` uses
one shared bracket encoder; every word of a sentence receives the same
relative index. Per-word objectives exclude that shared band. Sentence
addresses include source document, relative index and identified content;
`.when` alone does not distinguish documents.

*From `Architecture.md` lines 1721-2069, moved 2026-10-30.*

### Proposal: bounded pi and sigma folds as normalized means (Alec, 2026-09-21)

**Decided: dropped (Alec, September 25).** These means were evaluated in
real grammar folds and rejected. The normalization mixin, constructor
arguments, XML parameters and mode tests are removed. Native min/max perception
and the max concept pyramid remain the definition.

#### 1. Problem

The current SigmaLayer is tanh(W·atanh(x) + b) and the current PiLayer is
tanh((W·2atanh(x) + b)/2). These are the same map up to a factor of 2 in W
and b. Multiplying odds is adding log-odds, so pi and sigma operate in the
same chart, and any stack pi(sigma(...)) collapses to a single affine map
in the atanh chart followed by one tanh (verified numerically to 1e-16).
The network can only express a nonlinearity by driving values into the
clamp at ±(1 − eps), where the chart derivative 1/(1 − x²) explodes.
This is the root of both the wasted tanh/atanh pairs and the gradient
blow-up in the packed-row folds.

#### 2. Principle

A sigma-pi network gets its nonlinearity from the mismatch between the
chart where it multiplies and the chart where it adds. Therefore:

- sigma adds in the raw chart, with no entry or exit transform;
- pi multiplies in the log chart, and exits back to the raw chart.

Boundedness comes from making both folds *means*: every output lies between
the min and max of its inputs. A stack of means of any depth stays inside
the interval with no squashing and no clamp (only a log floor for pi).

#### 3. Definitions

All folds run on the perceptual chart u in [0, 1]. Conceptual space
x in [-1, 1] enters through u = (x + 1)/2 and leaves through x = 2u − 1.
Both maps are affine and exactly invertible; a signed product would give
parity rather than AND, so the rescale is not optional for pi.

Row-vector convention as in Layers.py: y = u @ W, so columns are outputs.

Weights. W = L D U as in NonNegativeInvertibleLinearLayer (softplus, so
every entry >= 0; init raw = −5 so W ~ I). Define the column sums
s = 1ᵀ W (a vector of length nOutput, s_j >= nInput · eps > 0) and

    W_n = W / s          (each column of W_n sums to 1)

W_n is a convex combination per output. It is invertible iff W is,
because W_n = W diag(1/s) and diag(1/s) is invertible.

Sigma (weighted arithmetic mean):

    forward:  y = u @ W_n
    reverse:  u = (y * s) @ W^-1          (the existing LDU solve)

Pi (weighted geometric mean):

    forward:  y = exp( log(max(u, EPS_LOG)) @ W_n )
    reverse:  u = exp( (log(y) * s) @ W^-1 ).clamp(0, 1)

Bias. An additive bias is what broke boundedness. Replace it with a convex
mix against a learned constant c in [0, 1] and a gate beta in [0, 1):

    sigma:  y = (1 − beta) * (u @ W_n) + beta * c
    pi:     y = (u-fold)^(1 − beta) * c^beta      (same thing in log chart)

with beta = sigmoid(raw_beta) and c = sigmoid(raw_c), init raw_beta = −5
so beta ~ 0 (no bias at init, matching today's near-identity init).
Reverse: u @ W_n = (y − beta c)/(1 − beta), then the solve above. Exact
because beta < 1 by construction.

#### 4. Invertibility and reconstruction

- The normalization is a rescale of the *weights*, computed from W alone,
  so it is a fixed positive diagonal and inverts exactly. It does not
  normalize activations and loses no degree of freedom in the data.
- Uniform averaging (W = 11ᵀ / n) is rank one and NOT invertible. That is
  why the fold must be a learned convex combination, not a plain average.
  Near-identity init gives an invertible start.
- Unary reverse: exact when y is in the image of the interval under W_n.
  When y is a reconstruction seed that no in-interval u produces, the
  solve can leave [0, 1]; clamp on the way out, as PiLayer2.reverse
  already does. No other guard is needed.
- Binary compose (chart parser): y = w_l * a_l + w_r * a_r per element
  (sigma) or the same in log chart (pi), with w_l + w_r = 1.
  Balanced split: a_l = a_r = y. The mean of equal operands is the
  operand itself, so generate() becomes the identity instead of
  tanh(s/2). synthesize_over_set: all M constituents = y.
  Known-reference split: a_r = (y − w_l a_l)/w_r, requires w_r > 0,
  which softplus guarantees; clamp to [0, 1].
- Gate on the LDU diagonal: unchanged. It scales columns before
  normalization; s is recomputed from the gated W.

#### 5. Sharpness (optional, same framework)

Means have no saturation, so crisp fits (XOR) may converge more slowly
than with tanh. If needed, both folds are members of the power-mean
family, which stays bounded for every exponent:

    M_p(u) = ( u^p @ W_n )^(1/p)

p = 1 is sigma, p -> 0 is pi, p -> +inf is max (OR), p -> −inf is min
(AND). One learned or scheduled p per layer buys sharpness without
leaving the interval. Reverse for fixed p: u^p = (y^p * s) @ W^-1, then
the 1/p root, clamped at 0 from below.

#### 6. Cost

Per element and layer today: atanh, clamp, matmul, tanh, plus a
singular gradient. Proposed: sigma is one matmul; pi is log, matmul,
exp. The reverse adds one elementwise multiply by s to the existing
solve. No custom autograd Function, no slope cap, no clamp inside the
grammar folds.

#### 7. Minimal implementation path

1. NonNegativeInvertibleLinearLayer: add a `normalize=True` mode that
   computes s = 1ᵀ W in compute_W_current and returns W / s; the
   reverse multiplies by s before the LDU solve. Gate flows as today.
2. SigmaLayer(nonlinear=False, invertible=True, monotonic=True,
   normalize=True): remove the atanh/tanh branches on this path; add the
   (beta, c) convex bias in place of the additive bias.
3. PiLayer: same inner layer; forward log -> matmul -> exp with the
   EPS_LOG floor; reverse as in section 3; drop _to_mult/_from_mult and
   _BoundedLogMult on this path.
4. generate / generate_functional / factorize_over_set: balanced split
   returns y; known-reference split per section 4.
5. Conceptual-space call sites: wrap with the [-1, 1] <-> [0, 1] affine
   rescale.
6. Tests: 30-deep alternating stack stays in [0, 1] without clamp;
   round-trip reverse(forward(u)) == u to solve tolerance; compose /
   generate round-trip; test_sigmapi and test_explicit_dimensions
   (crisp XOR) as the regression gate for section 5.

#### Evaluation (Claude, 2026-09-21)

Measured with small scratch scripts outside the repository, unseeded, every
run reported. The real-layer evaluation below repeats them.

**The algebra holds.** Sigma and Pi are one map in the atanh chart today, so
the diagnosis is right (the September 21 seed audit reached the same identity).
Weighted arithmetic and geometric means with convex weights stay inside
`[min, max]` of their inputs at any depth; `W_n = W diag(1/s)` is invertible
iff `W` is; the reverse formulas and the convex bias invert exactly; and the
rescale to `[0, 1]` is needed for pi.

**1. As written this is the monotone form, and that is its place.** Every
fold is non-decreasing in every input, for every power-mean exponent, and so
is any stack of them. That is what parts and wholes need — it is what keeps
parthood through a fold — and it is why this form cannot fit XOR (MSE .2500
in every run, a constant ½), and need not. That was the proposed placement before the perceptual fold layers were deleted (September 24);
the current native membership read remains min/max.

**2. XOR belongs to conceptual space, with the monotonic flags off** (Alec,
2026-09-21). Conceptual space needs no monotone operations, so its folds
take signed weights, and a negative weight *is* the complement: weight `−w`
on `x` is weight `w` on `not x`. No second rail is needed. What conceptual
space does need is that sigma and pi stop sharing a chart, and that they are
not *both* normalised: a mean has gain at most one, and a crisp XOR needs one
fold that can amplify. Inputs ±.9, targets ±1, eight runs each:

| signed pair in conceptual space | 2 hidden | 4 hidden |
|---|---|---|
| today: sigma and pi both in the log-odds chart | 0 of 8 (MSE 1.0) | 0 of 8 |
| pi → sigma in the raw chart, plain weights, tanh exit | 2 of 8 | 8 of 8 |
| **mean sigma in the raw chart → pi as it is today** | 6 of 8 | **8 of 8** |
| mean sigma → pi with normalised exponents too | 0 of 8 (MSE ≥ .63) | 0 of 8 |

The third row is the pair to try: **sigma** a signed, normalised combination
in the raw chart, with the convex bias toward a learned `c` in `[−1, 1]`,
inverting exactly as in §3 with `s` the column norm — feeding **pi
unchanged**, linear in the log-odds chart, where a signed exponent is a
literal (`m(−x) = 1/m(x)`) and the exit tanh bounds the result whatever the
weights. The gate should have at least four hidden units: at two it is
solved in six runs of eight, and a gate written at the minimal width invites
a seed again.

**Which norm: L2 for signed weights, L1 for the monotone means** (Alec,
2026-09-21: "signed weights will need an L2 norm, not an L1 norm (or both),
otherwise they might go unstable"). Measured on a signed near-identity
`L D U` (off-diagonals ±.0067):

| signed fold, column-normalised by | own-input weight at width 1032 | energy kept by one layer | gain of the reverse | 20 layers at width 264: energy kept / reverse gain |
|---|---|---|---|---|
| L1, `Σ|W|` | .13 | .13 | 10.8 | .0000 / 1.3 × 10⁹ |
| L2, `‖W‖₂` | .98 | 1.00 | 1.4 | 1.00 / 2.3 |

Under L1 the many small signed entries count in full in the normaliser and
cancel in the output, so the signal dies and its reverse explodes. L2 keeps
the energy and keeps the reverse tame, and it removes finding 4's width
problem for signed folds. Its bound is Cauchy–Schwarz, `|y| ≤ ‖w‖₂ ‖x‖₂`, so
it is exact on inputs of **at most unit energy** — which isolated normalized codes are. Actual grammar operands also include
composed values; their energy must be measured rather than inferred from
the codebook normalization. On cube-valued activations
(coordinates near ±1, energy above one) L2 alone overshoots into pi's clamp:
with XOR inputs of ±.9 per coordinate it was solved in 4 of 8 runs with a
third of sigma's outputs at the clamp, against 8 of 8 and none at the clamp
once the input vectors had energy .9. So: L2 for signed folds over unit-energy
vectors; L1, the plain column sum, for the non-negative means over
memberships; and where a signed fold must take cube-valued activations, both
— rescale to unit energy at entry, or keep the L1 bound there.

For the record, the monotone form can be made to fit XOR by giving it the
complement `[u, 1 − u]` and a learned power-mean sigma (`p` → 20): unsolved
with an arithmetic sigma (MSE .1669 every run; .125 is the best possible,
above the test's bar of .1), solved in 2 of 8 runs at two hidden units and
8 of 8 at four. That is not the recommended route — it puts negation into
the chart that does not carry it.

**3. The power mean must be computed in the log chart.** The literal
`(u^p @ W_n)^(1/p)` returned NaN in every run once an input sat at the floor.
`exp(logsumexp(p·log u + log W_n) / p)` is stable, and it is what the
complement runs above used.

**4. Near-identity does not survive normalisation at width.** With the
documented init (`raw = −5`, off-diagonals ≈ .007) the dense `L D U` product
has many small entries per column, and after column normalisation:

| width | own-input weight | spread kept by one layer | cond(W_n) |
|---|---|---|---|
| 8 | .95 | .96 | 1.1 |
| 64 | .66 – .70 | .68 | 1.5 |
| 264 | .23 – .36 | .27 | 4.4 |
| 1032 | .03 – .13 | .06 | 33 |

At production width one layer is already close to the uniform average the
proposal rules out as rank one. The off-diagonal init has to scale with
width (`raw ≈ −10` restores .95 at 1032), or the normalisation has to be per
node in butterfly mode, where every 2 × 2 node stays near identity whatever
the width. §7 step 1 should say which.

**5. The singular gradient moves; it does not vanish.** The atanh chart is
singular at both rails. The geometric mean is singular only at zero — but
zero is the common value of a sparse membership, and for a weight `w < 1`
the slope `w·u^(w−1)` is unbounded there. "No slope cap" needs checking in
the packed-row folds: a larger floor, or a cap at the floor only.

**6. Where inputs sit near ½, pi and sigma agree.** For `u = ½ + δ`,
`AM − GM ≈ (δ₁ − δ₂)² / 4`: second order. Coordinates of unit-norm codes at
production width are ≈ ½ ± .015, so folds over idea vectors stay close to
linear under this design, as they do under today's. The mismatch does its
work on memberships and activations, which span the interval. That may be
the right division — superposition is what keeps an idea decodable — but it
should be measured: the distribution of fold inputs in a trained model.

**7. A stack of means drifts to consensus.** Each layer's Jacobian is a
stochastic matrix, so spread and gradient both contract with depth (row 4
shows how fast when the weights are not near identity). The 30-deep test
should assert that spread survives, not only that the range holds.

**8. It does not help stored-idea generativity.** The balanced split returns
the parent for both children, which is what the reference-free inverse does
today. A mean is bundling: two codes are recovered from it by
correlation against the codebook, and their order only if the two weights
differ.

**9. Cost.** Sigma loses its chart entirely (one matmul and the bias mix);
pi goes from about nine elementwise kernels to about six; the custom
autograd function and the rail clamps go. The model is bound by kernel
launches, so that is where a gain would come from. Swapping one chart for
another, by itself, measured no faster.

**10. Where sigma alone builds higher orders, a normalised mean kills the
signal** (Alec's question, 2026-09-21). The concept pyramid is sigma only:
one hop per rung, a higher-order symbol from its members. A higher-order
symbol denotes a kind — any of its members — and activity is sparse, so usually
one member of several is active. A mean then gives the symbol the *share* of
its members that is active, and the shares multiply up the orders. With
equal weights and the leaf at 1:

| sigma at each rung, 8 members | one member active, orders 1 → 4 | all members active |
|---|---|---|
| normalised arithmetic mean (L1) | .125 → .016 → .002 → .000 | 1 at every order |
| L2-normalised, non-negative | .354 → .125 → .044 → .016 | 2.8 → 8 → 23 → 64 |
| power mean, `p = 20` | .90 → .81 → .73 → .66 | 1 at every order |
| max (`p → ∞`) | 1 at every order | 1 at every order |
| probabilistic sum `1 − ∏(1 − u)` | 1 at every order | 1 at every order |
| pre-11 tanh of the sum at weight 1 | .76 → .64 → .57 → .51 | 1 at every order |

At 32 members the mean is at .001 by the second order. The L2 form is worse
here: it still decays when one member is active and it grows without bound
when all are. So the arithmetic mean is the wrong sigma wherever sigma means
*union*. The laws this document already gives for sigma — identity 0,
absorber 1 — say the same: a mean has neither (`mean(x, 0) ≠ x`), while max
and the probabilistic sum have both, and neither decays. In the family of §5
that places the pyramid's sigma toward `p → ∞`. The probabilistic sum is the
other candidate, and it may be the cleaner monotone form: it is pi's De
Morgan dual, adding in the chart `log(1 − u)` where pi adds in `log u` — two
charts, so §2's mismatch holds — with non-negative exponents that need **no
normalisation** to stay in `[0, 1]`, and the same exact reverse through the
`L D U` solve. Its risk is the opposite of the mean's: many weakly active
members accumulate toward 1, which learned exponents below one and the
pyramid's top-K taper have to hold down. The arithmetic mean remains right
where an average is what is meant.

*Does pi build higher orders too?* The September 21 proposal predates native
perception (September 24).
Perceptual towers now use native min/max and have no learned fold layers;
subsymbolic passes retarget perception. Pi intersects located concepts only
in the order-0 field. Higher orders use max sigma over symbols of the
preceding order. Their sigma is a union that does not decay, and the store
rejects conjunctive edges above order 0. The comparison above does not
authorize an alternating pi/sigma symbolic pyramid.

**September 21 recommendation.** Evaluate behind a `normalize` mode, turned on
selectively. First the two XOR gates, in conceptual space with the monotonic
flags off: the signed mean sigma in the raw chart feeding today's pi, at
least four hidden units. Then the membership folds of the perceptual towers,
which already live on `[0, 1]`, comparing the monotone means as proposed
against the union forms of finding 10; the concept pyramid keeps a sigma that
does not decay. The grammar's idea-vector folds last, after finding 6 is
measured. Where a form
is adopted, the path it replaces is deleted rather than kept as a second
mode.

#### Real-layer evaluation: dropped (September 25)

Alec decided to **drop normalized means**. The experimental mode is deleted,
including its lift/lower arguments, configuration and mode tests. The
following measurements explain that decision; they describe the
evaluation source, not a callable production mode.

Signed L2 raw-chart Sigma followed by unchanged Pi solves all eight declared
unit-energy XOR runs at both widths 4 and 8, without reaching Pi's clamp.
The same-chart control and the both-normalized control solve none. On
cube-valued inputs the selective pair solves 0/8 at width 4 and 1/8 at width
8. Actual native grammar operands exceed unit energy: 25 of 68 operands in
34 retained binary windows, with a maximum norm of 1.7753 and a coordinate
of 1.0695. Normalized codebook leaves do not establish the required bound
for composed operands. No activation rescaling is added.

Width-scaled monotone initialization restores the own-input weight at width
1032 from .0486 to .9482, but 30 alternating arithmetic/geometric means keep
only about .166 of the input spread at width 264. The geometric floor has
zero gradient below it and a slope of 353.6 immediately above it in the
two-input probe. Convex bias and the log floor produce nonzero outputs on
all-zero membership input, conflicting with exact unknown evidence. They
are therefore unsuitable replacements for the native membership read.

The serial measurement ends at reconstruction .0891618710 with the current
folds and .0914028883 with selective normalization. Two timing pairs disagree
on the direction of the throughput difference; no consistent speedup is
shown. Balanced binary reverse reproduces the parent but cannot recover
distinct children without a known reference. Mean union decays through the
pyramid; max and probabilistic union preserve a single full member. The
historical projection limit of 256 positions (September 23) remains in that
comparison; under the located membership read (September 24) unrelated
membership evidence is exactly zero and has no such accumulation limit.

*From `Architecture.md` lines 2244-2346, moved 2026-10-30.*

6. **A pool of unminted concepts, learning all the time, with a gate on
   participation — in place of minting by significance** (Alec, 2026-09-22:
   "the minting of a term is difficult, since it may mean that we miss
   things that are not seen as significant, but they turn out to have large
   effect. I would prefer to have some number of 'unminted' concepts
   learning at any given time, with a gate that allows them to participate
   as concepts… almost like unassigned vectors in a VQ pool"). Concrete
   form, settled by the September 22 handoff:

   - **The pool becomes rows.** Each order keeps `P` provisional concepts
     in the store beside its minted ones (`P` a `model.xml` element). A
     provisional concept computes forward like any concept, but the next
     rung, the taper and the symbolic layer read it only through its
     **participation gate** `g ∈ [0, 1]`: they see `g · y`.
   - **Assignment, as in VQ.** By default (kinds): when two concepts'
     contexts match above a bar — the same where on different occasions —
     and no kind already holds both, the pair is assigned to the least-used
     provisional concept as its first two disjunctive parts; a concept whose
     context matches an existing kind's is added to it. With `<conceptualPi>`
     on (wholes as well): an incoming co-present set that no whole matches
     above the bar — cosine over conjunctive parts, as now — is assigned to
     the least-used provisional concept as its conjunctive parts, all of
     them, each necessary. Assignment is among unminted concepts only;
     minted concepts do not compete for input.
   - **Learning before minting.** A provisional concept's parts are tracked
     by EWMA — as the entry's member weights are now — and trained by the
     ordinary gradient through `g`, as the pyramid's weights are. Its
     context signature, the entry's context weights, is its where. This
     applies to what the mind witnesses: words and other percepts.
     Object-concepts are not witnessed, so their kinds are written by the
     closing from testimony, not discovered by the pool. The gate is **use,
     not gradient**: an EWMA of the concept's own activation being above a
     floor on recent inputs — earned by recurring, never optimised, so a
     concept cannot learn to participate by making itself cheap. `g` rises
     with use and decays without it.
   - **Where, for a concept, is its context — and that makes co-presence
     and substitution one rule** (proposal for Alec's question of
     2026-09-22: "We need to integrate the 'same space and time' that
     creates wholes and parts, and the 'substitution' that creates
     syntactic categories and parts of speech. I think we proceed by
     defining 'where' in terms of context, or perhaps by using the location
     as embedded in conceptual space (which reflects previous embeddings)").
     A percept has a `.where`, its span in the input. A concept has none;
     define its **where as its context**: what else is present in the field
     when it is — the leave-one-out bag the promotion observer already keeps
     as an entry's context weights, and the rotation updater already uses to
     place codes. Then one rule has two consequences:

     | | same time, different where | same where, different time |
     |---|---|---|
     | what it is | co-presence: parts of one whole | substitution: alternatives at one position |
     | fold | pi, a conjunction over what is present together | sigma, a union over what fills the same context |
     | in the towers | adjacent spans at one `.when`: a whole of its parts — its presence the conjunction of theirs, its extent the union of their spans (the synthesis fold) | one span position filled differently on other occasions: a type (WholeSpace's property rows) |
     | in the pyramid | with the switch on, the assigned co-present set: conjunctive parts, a whole | concepts whose contexts match: disjunctive parts, a kind — the default |
     | in linguistics | the syntagmatic axis | the paradigmatic axis (Saussure; Harris 1954) |

     Sigma over pi is then paradigmatic over syntagmatic: a concept is a
     union, over occasions, of conjunctions over what was present. The two
     definitions Alec offers are one thing at two time scales. The
     occurrence's where is its context *now*, and the code's location in
     conceptual space is its where *accumulated* — the rotation updater is
     the EMA that turns the first into the second, so "codes by
     distribution" is exactly substitution learned over time. Discovery uses
     the first within a pass (two concepts whose contexts match; with the
     switch on, a co-present set) and the second as the cue across passes (a
     nearby code is a candidate alternative). Geometry is the *evidence*; the edge is the
     *record*: a category is still membership, written when two concepts'
     wheres coincide above a bar, and a nearby code that is never seen in a
     matching context writes nothing. The category this discovers is the
     one the contexts distinguish: syntactic when the context is the frame
     ("the __ is": Mintz 2003; Redington, Chater & Finch 1998), semantic
     when it is the predicates ("__ purrs"), which is the distributional
     hypothesis at each grain. It applies to what the mind witnesses.
     Object-concepts have no witnessed where until there is a video feed, so
     their categories still come only from testimony; a word's discovered
     categories reach its object only through what language asserts.
     The context weights therefore stay in the pool: they are the where.
     The philosophical reading — particulars now, universals over time; the
     paradigm as the domain of exclusion; testimony for the unwitnessed — is
     in [Philosophy](Philosophy.md#where-is-context-particulars-universals-and-the-two-axes-2026-09-22).

   - **Discovery.** At `g ≥ θ_mint`, the
     concept is discovered: it gets its symbol and its code, LTM may
     reference it, and it is no longer recyclable. Use management is cleared
     and its gate is fixed at one, so its symbol survives non-use. Below `θ_recycle`, the
     least-used provisional concept is recycled when a new set needs one.
     Nothing is missed for being insignificant at first sight: every
     recurring substitution, and with the switch on every recurring
     co-present set, has a concept from its first recurrence, and an effect
     that shows late raises the gate then.
   - **Cost.** `P` rows per order in the store, dense within the pool's
     edges at first and pruned to the strong edges at minting; the host
     dictionaries go. The forward cost is that of `P` more rows per rung.
   - **The XOR gate.** Four or eight provisional conjunctions observe
     located primitive-input cases, without seeded selection. A sigma row
     learns which symbolized cases supply XOR. Its positive pole must match
     the truth table; a corner read cannot satisfy the gate.
   - **The switch, restated.** `<conceptualPi>` (default off) is whether the
     pyramid discovers wholes from co-presence as well as kinds from
     substitution: off, it is a taxonomy of kinds and every discovered
     concept is a union of sufficient parts; on, co-present sets also mint
     wholes of necessary parts, and the two stages of the computation both
     run.

*From `Architecture.md` lines 2348-2361, moved 2026-10-30.*

**Implementation and evidence (September 22–23).** The original
landing implements the union
replacement, both sparse part matrices, provisional rows and context weights,
transpose, checkpoint and optimizer ownership. The default remains sigma
alone; `conceptualPi` enables the conjunction stage. The
review correction implements the
paired representation, scoped dual folds, measured admission floor, dense
host contexts, permanent discovered gates and one sidecar owner for part
values. Rung and declared-part XOR checks establish composition. At those
commits, the two unseeded learning gates remained null and the serial
reconstruction baseline was unchanged; the trained live parallel measurement
records its weak-evidence limitation. The September 23 landing supplies learned
primitive properties and replaces the expected-failure learning assertions.

# STM.md

*From `STM.md` lines 23-40, moved 2026-10-30.*

> **2026-06-02 update (subsymbolic analyzer).** Operators no longer enter
> the STM idea space. They are kept in the SS **codebook**
> (`WholeSpace.insert_operations`, wired into `SymbolSubSpace.__init__`)
> and resolved as a soft superposition over the operator-prefixed parse
> tree; the STM idea slots hold only **combined meanings** -- an operator
> defines *how* meanings combine, contributing none of its own.

> **Status (2026-05-30):** new chapter for the STM serial / parallel
> modes work.
> Documents the per-batch STM buffer, the predict-then-perceive cadence
> (serial and parallel), the attentional-filtering regime (serial runs
> **with** attention by design — the old serial-vs-attention guard was
> lifted), the routing-parser SS-analysis / CS-execution split, the
> in-STM `IntraSentenceLayer` AR predictor, masked-word reconstruction
> via priming, the relative-vs-absolute end-state preservation with its
> grammatical clause closing, and the LTM chain of end-states feeding
> inter-sentence prediction. Where a piece is a deliberate scaffold or a
> deferred wiring, this chapter says so.

*From `STM.md` lines 472-541, moved 2026-10-30.*

## 6. IntraSentenceLayer

**Retired (recorded 2026-10-30).** `IntraSentenceLayer`, `intraLossWeight` and
`consume_intra_loss` no longer exist; the one predictor is `ActivationPredictor`
([ExpectationRetention](ExpectationRetention.md)). This section is the record.

`IntraSentenceLayer` ([Layers.py](../bin/Layers.py)) is the in-STM
autoregressive predictor — the layer that stakes the prediction in
predict-then-perceive. It is owned by ConceptualSpace (`self.intraSentenceLayer`).

**Architecture: combined PI-then-Sigma, no intermediate $\tanh$.**

- `self.pi`: `PiLayer(concept_dim $\to$ working_dim)`, `invertible=True`,
  `nonlinear=True` — the log-domain multiplicative boundary fold lifts
  each STM slot into the working width and bounds it to $[-1, 1]$.
- `self.sigma`: `SigmaLayer(working_dim $\to$ concept_dim)`,
  `invertible=True`, **`nonlinear=False`** — a raw linear $W x + b$ that
  collapses the lifted slots into the predicted idea.

The defining requirement is that there is **no extra activation
interposed** between the PI body and the Sigma body: `sigma` is built
`nonlinear=False` (a raw $W x + b$), so no additional $\tanh$ sits
between PI's output and Sigma's input. This is **not** a linear fusion —
`pi` is `nonlinear=True` (its symmetric log-domain $(1+x)/(1-x)$
embedding is an intrinsic nonlinearity), so the composite is a nonlinear
PI followed by a raw-linear Sigma, not two fusable linear cores.
`working_dim` defaults to `concept_dim`, keeping both sublayers square
isomorphisms so the parallel per-slot round-trip
$\text{reverse}(\text{forward}(x)) \approx x$ is exact up to the LDU
inverse tolerance.

**Forward signature.** `forward(prior_slots, routing=None, parallel=False)`
with `prior_slots: [B, K, D]`:

- **Serial collapse** (`parallel=False`, the primary regime): PI-lift
  every slot, **sum-fold over the slot axis**, then Sigma-collapse to one
  idea $\to$ `[B, D]`.
- **Parallel per-slot** (`parallel=True`): PI$\to$Sigma per slot, no
  cross-slot mixing $\to$ `[B, N, D]`.

The serial collapse is many-to-one (sum over $K$ slots), so its
`reverse` is necessarily approximate (it divides the recovered fold
equally across the $k = \text{stm\_capacity} - 1$ slots); the parallel
per-slot path is width-preserving and exactly invertible.

**Loss.** $\mathcal{L}_\text{intra} = \mathrm{MSE}(\hat{c}_t, c_t)$.
`IntraSentenceLayer.intra_loss(pred, target)` ([Layers.py](../bin/Layers.py))
is a plain `F.mse_loss` helper documented as the training-path wiring
point, but the live path does **not** call it: `ConceptualSpace.
_accumulate_intra_loss` ([Spaces.py](../bin/Spaces.py)) inlines the same
math (`(prediction - target).square()`) directly. `intra_loss` is
exercised only by `test/test_intra_sentence_layer.py`, not by any forward
call site — an unused-but-documented helper, not dead code to delete.

`_accumulate_intra_loss` keeps each per-word step's scalar loss in a
**list** rather than folding it into a running `accum + step_loss` sum
(2026-07-08 fix): the old running-sum pattern built a per-word add-chain
*inside* the compiled forward, which Inductor inlines transitively into
one kernel with a buffer argument per step — over Metal's 31-buffer
kernel-arg limit on wide configs (e.g. `N=64` produced a 34-arg
scalar-sum kernel, a fullgraph blocker immune to fusion caps). As a list,
each step's scalar stays an independent graph output, and
`consume_intra_loss` ([Spaces.py](../bin/Spaces.py)) chunk-sums the list
**eagerly** (post-body, pre-backward, mirroring the ARMA term) and
returns the per-step mean, resetting the accumulator. The weight is
`<intraLossWeight>` (default `0.1`); the term is gated off when grad is
disabled or the weight is non-positive.

---


*From `STM.md` lines 563-590, moved 2026-10-30.*

> **Routing conditioning of the intra predictor is LANDED.** Every
> `SymbolSubSpace.compose` call (per-word or boundary, default-only or
> full-router) builds a first-class `RoutingState`
> ([Language.py](../bin/Language.py)) ADDITIVE to the unchanged
> `current_rules` host-side rule dict: `_synthesize_rule_probs`
> ([Language.py](../bin/Language.py)) turns the fired rule_ids into a
> dense `rule_probs: [B, n_rules]` distribution — a gradient-bearing
> soft-marginal aggregation (`_synthesize_rule_probs_soft`) whenever the
> router ran tensorially, or a DETACHED hard scatter
> (`_synthesize_rule_probs_hard`: unit mass onto the fired rule-ids,
> L1-normalized per row) on the default-only fast path — and stashes it
> on `symbolSpace.routing_state.rule_probs`.
> `ConceptualSpace._intra_routing_for_predict`
> ([Spaces.py](../bin/Spaces.py)) reads that tensor, returns it only when
> its last dim matches `n_rules == len(TheGrammar.rule_table)`
> (`IntraSentenceLayer.routing_proj`'s expected width) and aligns its
> batch dim to the predictor's STM-snapshot context (broadcast when one
> side is `B=1`; `None` on an unreconcilable mismatch, fail-loud on a
> non-finite tensor). Both `_stm_predict_then_perceive_serial` and
> `_stm_predict_then_perceive_parallel` ([Spaces.py](../bin/Spaces.py))
> pass the resolved routing tensor into `intraSentenceLayer.forward`,
> which projects it `[B, n_rules] \to [B, concept_dim]` via
> `routing_proj` and adds it as a bias to the Sigma output
> ([Section 6](#6-intrasentencelayer)) — so the per-word fire now DOES
> make the in-STM predictor rule-aware, not just the SS dispatch context.
> `routing=None` (no reachable `symbolSpace`, wrong width, or an
> unreconcilable batch mismatch) degrades to the un-biased predictor,
> byte-identical to the pre-wiring behaviour.

*From `STM.md` lines 827-867, moved 2026-10-30.*

### Structured production path

`sentenceExpectationScope=structured` is the model default when sentence
prediction is enabled. `SentenceExpectation` reads the bounded
chronological observation view as `[K, 3, D]` values plus `[K, 3]` occupancy.
The three positions are NP1, VP and NP2. Padding is masked before the network;
role and chronological positions remain distinct. It predicts three separate
vectors, three presence logits and one clause-kind logit. The loss combines
role MSE with binary cross entropy for presence and clause kind. Targets are detached;
current-step source representations remain live under the objective-local
boundaries in [GradientFlow](GradientFlow.md). For individual-reference
grammars, the latest source is re-encoded through the current identity
columns before each trial. Older context and the observation target detach.
Once columns exist, role prediction error is measured in their coordinates;
presence and clause-kind losses keep their existing meanings. This preserves
sentence optimizer boundaries without keeping a preceding compose graph
across an update. See [Layers.py](../bin/Layers.py) and
[IndependentComponents.py](../bin/IndependentComponents.py).

The independence population comes from retained external observation rows,
excluding definitions, estimates and asserted facts. It is snapshotted before
reading, independently of the batch partition. Accumulated innovations use
identified object coordinates and retain occurrence addresses. A has at most
two active noun columns per ordinary frame; B has at most one change column.
The same expectation objective owns both dictionaries. Recurrence admission
and witness counts happen only after the kept closing. These are mechanism
contracts under review, not a finding of learned identity or verb reuse.

The packed observer uses the existing ended end-slot/depth outputs for each
sentence, with an explicit STM-to-infix permutation. Corpus source addresses
identify each sentence's document, including changes within a packed row.
`begin_document` clears only the selected prediction stream; already-scored
losses, other rows and durable history survive. Soft packed-brick resets keep
the same document stream; only hard resets or document changes make it cold.
Weight restore starts the
observation view cold. These are sequence and local-role contracts; retained
compound references remain part of the separate nesting migration.
See [Layers.py](../bin/Layers.py) and
[Models.py](../bin/Models.py).


*From `STM.md` lines 868-956, moved 2026-10-30.*

### Historical root baseline (`sentenceExpectationScope=root`)

A **lifted `IntraSentenceLayer` instance** (`_inter_predictor`,
[Layers.py](../bin/Layers.py)) predicts the next end-state over the
external observation sequence — the same predictor class as the in-STM one,
instantiated at the inter-sentence level. Its transient `_inter_context`
view is bounded and per-row; durable LTM retains detached observations.
Current-step source context keeps its encoder graph. Consuming the prediction
loss and entering the next brick detach that view without deleting history.
Document resets clear the selected view and pending estimate, preserving
other rows and durable LTM ([Layers.py](../bin/Layers.py),
[Layers.py](../bin/Layers.py),
[Layers.py](../bin/Layers.py)). Its chain window is
$K = \min(\text{ltmCapacity}, 8)$ (`_inter_chain_window`): the AR signal
that predicts the next end-state lives in the last handful of sentences,
so a small bounded window is used rather than the full `ltmCapacity`.

`predict_next_end_state(b=0)` ([Layers.py](../bin/Layers.py))
produces the next end-state **shape** $(\hat{d}, \hat{p}[\hat{d}, D])$:

- **Chain reduction (ragged $\to$ fixed), mode-dependent.**
  Each chain entry is reduced to its **root** by
  `_reduce_end_state_to_root` ([Layers.py](../bin/Layers.py)) — but WHICH
  slot is the root depends on the LTM mode ([Section 10](STM.md#10-ltm-as-the-chain-of-stm-end-states)):
    - **Legacy mode.** The end-state is stored newest-first, so the root
      lives at the **last** slot ($\text{depth}-1$): for an absolute
      end-state (depth 1) that is slot $0$ (the collapsed idea); for a
      relative end-state (depth 3) it is slot $2$, the predicate (the head
      the relative structure hangs off).
    - **Consolidated mode.** The payload is the store's native INFIX
      `[idea1, predicate, idea2]` order, and the root is always **slot
      $0$** = idea1 — the subject/topic, present even when there is no
      predicate (an absolute row has no `predicate`/`idea2`, only
      `idea1`), so it is the one slot every row shares regardless of
      depth.

  The last $K$ roots form a `[1, K, D]` context, left-padded with
  zeros so the most recent sits at the tail (newest-at-$-1$, like the ARMA
  ring).
- **Root prediction.** `_inter_predictor.forward(context, routing=None,
  parallel=False)` $\to$ `[1, D]`, the predicted root.
- **Loss.** $\mathcal{L}_\text{inter} = \mathrm{MSE}(\hat{p}, p)$ on the
  roots, accumulated by `_accumulate_inter_loss`
  ([Layers.py](../bin/Layers.py)) and drained by
  `consume_inter_loss`, weight `<interLossWeight>` (default `0.1`). The
  actual root is detached for this comparison, while live preceding context
  can train the encoder as well as `_inter_predictor`. Teacher reconstruction
  does not disable this term. `observe_stm_end_state` scores each arriving
  observation once. Evaluation does not accumulate training losses. Joint
  gradients use the [objective-local gradient contract](GradientFlow.md).
- **InfoNCE next-idea contrastive term (optional, additive).** When
  `<interContrastiveWeight>` is positive (default `0.0`, off),
  `observe_stm_end_state` also ranks the actual next root above the
  chain's past roots (negatives) under $\cos(\hat{p}, \cdot) /
  \text{temp}$ (`<interContrastiveTemp>`, default `0.1`) via
  `_accumulate_inter_contrastive` $\to$ `consume_inter_contrastive_loss`
  ([Layers.py](../bin/Layers.py)) — a `torch.nn.functional.cross_entropy`
  over `[pos_root; neg_roots]` with the positive at index 0. Best-effort:
  a short/odd chain simply yields fewer negatives (the accumulator no-ops
  with none; the MSE term above still runs regardless). Fail-loud on a
  non-finite step.

The prediction is subtracted only at the closing, as described in
[ExpectationRetention](ExpectationRetention.md). The observation remains in STM
and LTM unchanged; the conceived roles are derived evidence for the chooser.
`generate_sentence` sends a positive predicted idea directly to `<generate>`.
There is no comprehension-time prediction injection.

The following depth-copy behavior describes the explicit **root benchmark**;
production structured expectation predicts three distinct vectors and presence.

> **Honesty — $\hat{d}$ is a copy-last AR prior.**
> The predicted **depth** $\hat{d}$ is a simple AR prior: the depth of the
> **most recent** end-state in the chain (a relative sentence tends to be
> followed by structure of the same shape; an absolute by an absolute).
> This delivers `depth in {1, 3}` without a separate learned head, but it
> is a **scaffold** — a tiny `concept_dim $\to$ 2` argmax head is the
> documented upgrade path. The chain reduction always collapses each
> end-state to ONE root vector rather than mean-pooling over depth (the
> root carries the sentence-level signal, mirroring the ARMA
> `_pool_sentence_rep`) — but which physical slot is "the root" is
> mode-dependent (see the "Chain reduction" bullet above): it is
> "slot-0" only in consolidated INFIX mode, where idea1 is always at
> slot 0; in legacy mode the root is the OLDEST slot
> ($\text{depth}-1$), which is slot 0 only for a depth-1 absolute
> end-state. A non-finite predicted root raises.

---


*From `STM.md` lines 957-1006, moved 2026-10-30.*

## 12. nanochat comparison framing

The trainable target is `MentalModel.xml`:
`<symbolicOrder>1</symbolicOrder>`,
`<data><dataType>embedding</dataType>`,
`<sentenceExpectation>true</sentenceExpectation>`,
FineWeb data (`<shardDir>data/fineweb</shardDir>`). This is the
configuration that exercises the full STM stack — serial sequencing, the
CS$\to$PS windowing and CS$\to$SS taxonymic mask
([Section 4](STM.md#4-attentional-filtering)), the in-STM and inter-sentence
predictors, and the LTM chain. It does **not** set `<attention>`, so that
knob resolves to its default `"off"` — see the correction in
[Section 4](STM.md#4-attentional-filtering); the legacy `<hasAttention>` boolean
is deprecated and inert regardless.

The comparison loss, as trained by `runBatch`, is

$$
\mathcal{L} = \mathcal{L}_\text{IR} + \mathcal{L}_\text{intra}
+ \mathcal{L}_\text{inter} + \mathcal{L}_\text{ARMA}
+ \mathcal{L}_\text{inter-contrastive},
$$

the masked-LM information-reconstruction term at the subsymbolic (PS) (see
[Spaces.md](Spaces.md#within-sentence-ar-retirement)) plus the
in-STM next-idea term ([Section 6](#6-intrasentencelayer)) plus the
inter-sentence next-end-state term ([Section 11](STM.md#11-the-recent-situation))
plus the discourse ARMA($p$, $q$) term (`BasicModel._discourse_arma_loss`,
[Models.py](../bin/Models.py); `InterSentenceLayer.observe`,
[Architecture.md](Architecture.md) — a separate ring-based sentence-rep
predictor on the SAME `InterSentenceLayer`, distinct from the STM-chain
`_inter_predictor` of Section 11) plus the optional InfoNCE next-idea
contrastive term ([Section 11](STM.md#11-the-recent-situation), off by
default via `<interContrastiveWeight>`). This is analogous to a
transformer language model's next-token cross-entropy, but computed **in
concept space** rather than over a token vocabulary:
$\mathcal{L}_\text{intra}$ predicts the next idea within a sentence,
$\mathcal{L}_\text{inter}$ the next sentence's end-state shape,
$\mathcal{L}_\text{ARMA}$ the next sentence-level representation over the
discourse AR/MA rings, and $\mathcal{L}_\text{IR}$ reconstructs masked
content — together the concept-space counterpart of the autoregressive LM
objective a system like nanochat trains.

> **Out of scope.** The comparison harness itself — running this against a
> transformer baseline and reporting the numbers — is a **separate plan**.
> This chapter documents the loss that *would* be compared and the
> configuration that produces it; it does not build the benchmark.

---


# Spaces.md

*From `Spaces.md` lines 1058-1069, moved 2026-10-30.*

### AR cursor unfold retirement (2026-05-13)

The legacy AR-training path padded + unfolded the embedded sentence
into `[B, K, N, D]` cursor windows so the body could see a
`[B*K, N, D]` parallel view of every prefix. At `bs=128`, `K=128`,
`N=1024`, `D=10`, the unfolded tensor alone was ~320 MB.

The unfold was retired for AR training on 2026-05-13, replaced with a
serial K-cursor loop (`_forward_per_stage_no_unfold`) that walked
the same prefixes with a `[B, N, D]` tensor and a per-cursor causal
mask.


*From `Spaces.md` lines 1428-1443, moved 2026-10-30.*

## SyntacticSpace --- retired

The standalone `SyntacticSpace` class has been retired. LanguageSpace owns
one operation-selection layer for compose, the serial STM reducer and the
shared numerical grammar catalog. It chooses one binary or unary operation
per round, with STOP eligible only when the sentence fits its row.

WholeSpace supplies property memberships and has no word dictionary or
symbolic compose path. The temporary reading record supports reconstruction
and the forced explore deviation while a sentence is open. A completed row
stores its one-slot or three-slot end state; readback generates from that state.

See [Language.md](Language.md) for grammar and parser dispatch.

---


# Language.md

*From `Language.md` lines 498-520, moved 2026-10-30.*

## STM Shift/Reduce

`ShortTermMemory.shift` / `.reduce_step` / `.reduce_step_soft` and the
`_RuleScorer` MLP (`bin/Layers.py`) are a typed admissibility-masked
shift/reduce driver — SHIFT snaps an input vector to the nearest live
reference and pushes `(payload, category, order, ref_id)`; REDUCE masks
rule logits by typed admissibility, softmaxes over what's admissible,
and records the argmax. It has **zero production callers**: every call
site is `test/_stm_test_fixtures.py`, which keeps the surface alive as a
compat shim for tests written before the 2026-05-21 SymbolSubSpace
refactor. The live parsing mechanism is `LanguageLayer.compose`'s
single softmax over operations and locations; see [STM.md Section 5](STM.md#routing-parser)
for the accurate, audited account of SS-analysis vs CS-execution.

`ConceptualSpace.stm` (a `ShortTermMemory` instance) is, in the live
model, a plain payload/idea stack, not a typed parser stack: its
per-batch slab and depth pointer (`_buffer` / `_depth`) are ordinary
tensors written by `push_step_masked(ideas, gate)` (the gated
newest-at-slot-0 push) and read back by `snapshot()`. It carries no
category / order / ref-id typing — that typed buffer is what the dead
shift/reduce driver above expects, and nothing in the live model
populates it.


*From `Language.md` lines 604-653, moved 2026-10-30.*

### Soft Operator Superposition

`operator_superposition(query_vec)` is a softmax over the cosine
similarity between a query and every operator vector.
`soft_operator_compose(dist, left, right)` is the distribution-weighted
sum of each operator's `compose`, evaluated per operator *arity* ---
unary operators such as `exist` are called as `compose(a)`, binary
operators as `compose(a, b)`. A one-hot distribution reduces exactly to
the typed grammar; a spread distribution superposes operators, and that
superposition is the mechanism that discriminates `A AND B` from
`A OR B` while a single layer is chosen.

`shape_operators(examples, op_names, steps, lr, seed)` trains the
operator vectors by backpropagating an MSE between the superposed
prediction and the supervised result, writing the shaped vectors back
into `_operation_vectors`. The op set may span the whole grammar, so the
shaper filters to layers of arity 1 or 2 and dispatches `compose` by
arity.

**Status (D1 gate --- met).** Role-collapse does not swap declared POS for
another single-label POS system; it replaces declared shared categories
with operator-local participation categories that a word may fill several
of (overlaps are expected). The D1 gate is therefore not a single-label POS
recovery test --- it asks whether those participation patterns are
*structured enough to drive a learned collapse* into the smaller
mutually-exclusive category set the live parser needs. They are:
`participation.learned_collapse` proposes merges by participation similarity
and accepts only those that keep every grammar rule distinguishable
(`collapse_conflicts == 0`); on the transitional `complete.grammar` it
compacts 43 context-unique symbols into 14 mutually-exclusive categories
with zero parser conflicts (`test/test_d1_pos_recovery_gate.py`). The exact
substitutability congruence is trivial there (every symbol is
context-unique), so "recovers the grammar" means the parser's rule
decisions survive the collapse, not exact rule regeneration. With the gate
met, the former standalone role-collapse file has been absorbed into
`complete.grammar`, which is the broad live role-only grammar used by
`MentalModel.xml`. The part relation is unified there: role-labelled structural
faces declare `part` and its converse `whole`, and the loader derives their one
canonical thought-operation family. There are no separate boundary declarations
and no `query` rule attribute. A `what` structural wrapper marks a completed
idea interrogative; only the completed-row boundary dispatcher can execute it.
[Relation declarations](../data/complete.grammar#L99),
[thought contract](QueryContracts.md).

The historical `_SURFACE_TO_KIND` aliases remain an isolated compatibility
adapter for the older reasoner. They do not define a production grammar
operator, native VP, selected thought action, or learned feature.
The declared thought catalog and participation clustering are live and tested.
Production compose uses the hard derivations described below.


*From `Language.md` lines 898-941, moved 2026-10-30.*

## Future work: nouns from PartSpace, adjectives from WholeSpace

> **Mostly superseded (Alec, 2026-09-28).** The opening of this section is
> now decided in direction: a noun is the same kind of function as an
> adjective, already mapped over the top domain, and that top domain is the
> domain of discourse at its widest
> ([above](Language.md#words-narrow-the-domain-of-discourse)). The
> conjecture that follows it — concrete nouns sourced from PartSpace and
> adjectives from WholeSpace, so that part of speech falls out of the pole a
> symbol is pre-applied to — is "an old conjecture, mostly undone by the
> hypothesis that words are projection operators onto lower-dimensional or
> smaller subspaces". Part of speech is a role in a derivation. The section
> is kept for the record.

The signed-space snap models a concrete
noun as an adjective pre-applied to the top domain --- `black(cat($\ldots$))`
treats "cat" as the same *kind* of function as "black", the later one
already mapped over $\mathbb{1}$. That modeling choice is probably already
latent in the mereology poles. The lattice poles are vectors of the
presence domain (see `doc/Architecture.md`): **NOTHING** $= [0,0,\ldots]$
is the part/bottom pole, **EVERYTHING** $= [1,1,\ldots]$ is the whole/top
pole. So:

- a **whole (property)** is pre-applied to EVERYTHING --- narrowing the
  wide-open $\mathbb{1}$ object downward (the `assert_concept_relation`
  first-concrete-whole-replaces-EVERYTHING move); this is the *adjective*
  shape, a modifier that carves the domain;
- a **part (particle)** is pre-applied to NOTHING --- building presence up
  from $\mathbf{0}$ (the first-concrete-part-replaces-NOTHING move); this
  is the *concrete-noun* shape, an object accreted from constituents.

The conjecture for a future pass: source **concrete nouns from PartSpace**
and **adjectives from WholeSpace**, so the grammatical part-of-speech
distinction falls out of which pole a symbol is pre-applied to, rather than
being a learned category. This would give the snap's ADJ(N) intersection a
principled home --- the noun's broad PS-side support and the adjective's
narrowing WS-side mask are then *typed by origin*, and the
support-restricted metric (score over the parent's support, unaffected by
the modifier's attenuation) becomes the natural recovery law rather than a
tuning choice. Open questions: how order-raising (`maybe_raise_order`)
interacts with a PS/WS part-of-speech split; whether abstract nouns want
the WholeSpace (property-like) or PartSpace (object-like) origin; and how
the `<Anchors>` technical operator spellings sit relative to this axis.


# Componentization.md

*From `Componentization.md` lines 154-160, moved 2026-10-30.*

> **Packed-root scan experiment (2026-07-26).** Replacing the detached
> `[B,S,D]` sentence-root FIFO with a per-word `torch.scan` output preserved
> roots bit-for-bit, but PyTorch 2.14's prototype scan increased saved
> autograd state from 720,900 to 2,742,395 bytes on the W16 canonical probe
> (3.8×). A same-width integer bit-transport probe also left peak memory
> unchanged, confirming that AOTAutograd already excludes the detached FIFO
> from the effective differentiated tape. Neither variant is retained.

# SymbolFirewall.md

*From `SymbolFirewall.md` lines 137-152, moved 2026-10-30.*

### A. MetaSymbol participation categories — *a typed symbol owns a latent

*Record (audit 2026-10-30): `MetaSymbol` and the category codebook no longer exist in the code; promotion is MTM recurrence. The section is kept as written.*

category* (invariants #3, #7)

A word's syntactic category is learned as its **frequency of participation across
operator roles** (`<method>_I<n>` inputs / `<method>_O1` outputs), stored as a
small `VectorQuantize` codebook keyed by the **MetaSymbol** (the equivalence
class uniting a word with its object). The latent category is therefore *owned*
by an addressable symbol — not floating in a residual stream — and recurrent
role profiles are *promoted* into stable centroids that decay when unused
(`codebook_retire=False`). The structured grammar layers condition routing on
this typed category context through `<categoryCodebook>` for all transform
choosers. Design: [Language.md](Language.md) "Participation Categories".


*The whole of `NanoChatGrammarPilot.md` (263 lines), moved 2026-10-30.*

# NanoChat-sized grammar pilot



> Historical gate record. These fixed W=64 comparison fixtures predate the
> current `data/BasicModel.xml` architecture. Use `basicmodel.txt` and
> `data/BasicModel.xml` for current training; retain this cohort only for
> reproducing the original held-out evaluation.

## Question and falsifiable first milestone

The comparison-sized experiment asks whether the serial grammar architecture
can acquire predictive language structure from FineWeb-Edu, measured against
Karpathy's default NanoChat speedrun tier, depth 20 (Alec, 2026-09-25;
replaces the earlier matched-parameter depth-4 reference). The two are not
parameter-matched: the comparison is by bytes seen and wall time on the
common held-out item set, and the depth-20 result is the ceiling reference —
what the default $100 speedrun does on these items — not a matched curve.

The first milestone is deliberately narrower:

> After training only on the train documents, use the preceding words to pick
> the true next word from 16 surface-word candidates in held-out FineWeb-Edu
> documents.

For BasicModel, the evaluator uses the shared expectation owner's
**word-level categorical distribution**, conditioned on the observed prefix.
All 16 candidates share one bank and are scored by negative log probability.
Candidate completions never enter the predictor's prefix history. NanoChat
scores its same candidates by conditional log probability. Top-1 and reciprocal
rank remain directly comparable; this finite-bank likelihood is not token
perplexity.

**Scope clarified by Alec, 2026-10-03:** the check is of the evaluator's mechanism
on a small configuration over a few unchanged manifest items. The language gate
waits for a trained checkpoint. It does not call for scoring the fresh
BasicModel now.

The evaluator stages the normal native word readings, then scores the complete
candidate bank under `no_grad`. It does not build sentence composition or the
answer output, which cannot change the staged word snapshot. Tests compare its
scores exactly with the full-forward boundary. Native evidence and priming
omit inactive padding during eager staging; the returned word buffers retain
their configured dimensions. No item, candidate, or live word is removed.

This is a better language gate than reconstruction loss.  An invertible model
can learn to copy an ill-formed sentence, and its raw MSE is not a normalized
language likelihood.  The multiple-choice task requires context to identify an
unseen held-out word.  A shuffled-prefix control tests whether word order, not
only unigram frequency, supplies the gain.

Use 500 fixed held-out items.  Each has a configured 4--16 word prefix, the true
next word, and 15 frequency/length-matched distractors sampled only from the
held-out documents.  The checked-in realization has 4--7 word prefixes because
it was frozen while the prototype still spent one flat 32-percept slab across
the sentence.  Keep it frozen for comparison, but that restriction is no longer
architectural: the corrected model has an independent 64-word sentence axis and
set-synthesizes every raw constituent inside that word's iteration. PS, WS, CS,
and STM remain eight wide/deep. The evaluator rejects an item if its candidate
falls beyond the outer word cap or if raw-constituent staging ever reports a
cut (which is now an implementation failure, not a PS-capacity condition). The
initial gate is:

- top-1 at least 12.5% (twice the 6.25% chance rate),
- at least 5 percentage points above the same initialized model,
- shuffled-prefix accuracy below intact-prefix accuracy, and
- the result repeats for three seeds before scaling the model.

Also record candidate-bank negative log likelihood, candidate mean reciprocal rank,
and the variance/effective rank of target ideas.  Those last checks catch a
latent-collapse shortcut.  Exact surface reconstruction is useful smoke-test
telemetry, but is not a language pass criterion.

## Two model tiers

| Tier | Config | Parameters | W / F / D | Percept rows | FineWeb-Edu documents |
|---|---|---:|---:|---:|---:|
| Architecture gate | `data/MM_nanochat_grammar_gate.xml` | 8,518,478 | 64 / 8 / 256 | 8,192 | 200 |
| Comparison pilot | `data/MM_nanochat_grammar_pilot.xml` | 41,679,182 | 64 / 8 / 512 | 32,768 | 2,000 |
| NanoChat reference | Karpathy's default speedrun, depth 20 | about 560M (per the nanochat defaults; record the exact count from the run) | context 2048 / -- / 1,280 | 65,536 BPE vocab | the speedrun's own FineWeb-Edu shards, about 11B tokens |
| Existing FineWeb config | `data/MM_20M_fineweb.xml` | 118,256,431 | legacy 64 / -- / 1,024 | 65,536 | 10,000 default |

Counts include the bounded STM reducer that is registered during the standard
pre-training warmup. The pilot is roughly one-thirteenth the size of the
depth-20 reference and sees roughly one five-hundredth of its bytes; the
comparison is deliberately unmatched, and the numbers are reported with
those ratios beside them. The old "20M" name is not a reliable budget: its
current live graph is about 118M parameters. Nothing NanoChat has been built
locally yet: the reference tier is trained on rented hardware (eight H100s,
about four hours for the depth-20 speedrun), and the evaluator's NanoChat
scorer — conditional log-probability of each of the 16 candidates given the
prefix, top-1 and reciprocal rank on the same item file — is still to be
written.

Here `W=64` is only the outer serial-loop bound. `F=8` is the simultaneous PS,
WS, and CS field width and the STM workspace depth. Raw radix width is not a
model capacity: at iteration `w`, PartSpace gathers `[B,P_raw,D]` for the
complete current word and applies its configured sigma set-fold. PS and WS then
bind in CS; one current-word concept, carrying its derived order, enters STM.
Concepts remain a
growing relation inventory, commonly one-part/one-whole identities, rather than
being capped at the instantaneous 8+8 percept field.

Both tiers therefore use:

- `subsymbolicOrder=4`, `symbolicOrder=4`, uniform eight-wide conceptual
  stages, and unbounded `syntacticOrder=0`;
- serial word traversal and `mereologyRaise=true`;
- the learned MLP transform chooser and category codebook;
- one routing pass (`learning=false`) for the first gate; and
- reconstruction, intra-sentence prediction, inter-sentence prediction, and
  contrastive discourse losses.

The tier table and configuration choices above are historical. Current runs
use `data/BasicModel.xml`: the mandatory paired grammar walks, one attention
budget and level-indexed expectation replace the old mode and order switches.
The frozen 500-item manifest and all numerical learning bars remain unchanged.
No fresh-model gate is scheduled.
Qualification still requires the million-sentence prerequisite and
ordered/shuffled/context-free controls at equal updates for the three
predeclared seeds.

## Corpus ladder

All counts below come from the currently downloaded
`data/fineweb/shard_00000.parquet`; the loader splits whole documents
deterministically 8/1/1 into train/validation/test.

| Stage | Documents | Sentences | Whitespace words | Purpose |
|---|---:|---:|---:|---|
| Launch check | 20 | 769 | 13,090 | finite forward/backward and obvious overfit |
| Language gate | 200 | 8,452 | 141,228 | held-out next-word retrieval |
| Comparison pilot | 2,000 | 89,189 | 1,504,570 | first NanoChat-sized learning curve |

Do not move to 2,000 documents because training loss merely decreases.  Move
only after the 8.52M gate beats initialization, chance, and shuffled context on
the fixed held-out item manifest.

The frozen evaluator now lives at `bin/eval_nanochat_grammar.py`; its checked-in
manifest is `data/eval/nanochat_grammar_gate.json`.  The manifest contains 500
items from all 20 test documents in the 200-document gate, has randomized answer
positions, and is pinned by item SHA-256
`ad594aa33ae1ac2f0bc49cc16c2822144fe26c73897f6b17228f641abc9773d7`.
`train.py --test` remains reconstruction telemetry; use this evaluator for the
language claim once a trained checkpoint exists. The commands below
are historical reproductions, not instructions to run a fresh gate:

```sh
cd basicmodel
.venv/bin/python bin/eval_nanochat_grammar.py generate

# Deterministic initialized baseline; explicitly bypass any XML autoload file.
.venv/bin/python bin/eval_nanochat_grammar.py score \
  --device mps --fresh --item-batch-size 32 \
  --output output/nanochat_grammar_gate_stm8_initialized.json

# Trained model. The reducer is prewarmed before checkpoint loading so all
# learned reducer/router weights are restored rather than ignored as lazy keys.
.venv/bin/python bin/eval_nanochat_grammar.py score \
  --device mps --checkpoint output/MM_nanochat_grammar_gate_stm8.ckpt \
  --item-batch-size 32 \
  --output output/nanochat_grammar_gate_stm8_trained.json
```

All prior initialized/trained results and checkpoints used either the flat
sentence-percept axis or the rejected 64-deep STM. They are diagnostic only and
must not be compared with this model. The current trained-checkpoint comparison
and its controls wait for a trained checkpoint.

Before training or scoring, run the explicit reduction audit:

```sh
.venv/bin/python bin/eval_nanochat_grammar.py trace --device mps --words 64
```

Add `--timeline` only when the per-word confidence/depth sequence is needed;
the default audit prints the compact aggregate.

The audit counts a reduction only when STM depth decreases and separates soft
occupancy-pressure decisions, hard capacity demands, grammatical boundary
closure, and any unlicensed fallback. The controller removes the binary-rule
count prior before deciding SHIFT versus REDUCE, then lowers the configured
`0.75` threshold as STM fills. At full depth it demands the best grammatical
operator; absence of such an operator is an error rather than a dropped word.

On a fresh MPS build, 64 pushes produced 61 soft-pressure reductions, zero
capacity demands, two grammatical boundary reductions, and zero unlicensed
reductions. Peak depth was 4 and final depth was 1, so all 63 required depth
decreases were accounted for inside STM8. The fresh operator mix (60 online
`conjunction`, one online and two boundary `disjunction`) is initialization
telemetry, not evidence of learned grammar diversity.

The depth-20 reference trains on the speedrun's own shards, not on the
BasicModel manifest. Two consequences. First, **contamination**: the speedrun
consumes the first FineWeb-Edu shards, and the checked-in gate draws its
held-out items from `shard_00000`; regenerate the item manifest from a shard
outside the speedrun's shard list before scoring the reference, or state
plainly that the reference has seen the test documents. Second, the
**matched curve is a separate, cheap run**: a small NanoChat (depth 4 to 8)
trained on the exact BasicModel document manifest exported to NanoChat train
and validation parquet files, minutes on one H100, reported beside the
depth-20 ceiling. Budget every run by source UTF-8 bytes seen and wall time,
not optimizer steps or tokenizer tokens.  Report parameter count, peak
memory, bytes/second, top-1/MRR on the common word-choice set, and learning
curves.  Do not compare BasicModel's reconstruction MSE to NanoChat
bits-per-byte or cross-entropy as though they were the same quantity.

## ArborStudio MPS launch posture

The corrected architecture runs locally on `ArborStudio.local` from
`/Users/arogers/github/WikiOracle/basicmodel`; MPS tensors are on `mps:0`. The
command-approval boundary does not move execution to another host or checkout.

Fresh measurements with the independent 64-word axis, eight-wide fields,
STM8, rule-count-neutral pressure controller, and no checkpoint load/save are:

| Gate operation / device | Result | End-to-end wall time |
|---|---:|---:|
| 64-word reduction audit / MPS | peak depth 4; 63 grammatical reductions; 0 unlicensed | 5.2 s |
| One optimizer update / MPS, batch 1 | reconstruction loss 0.5954 | 7.87 s |

These are launch checks, not learning results. The optimizer smoke used 20
FineWeb-Edu documents, stopped after one batch, and wrote no checkpoint. Batch
4 remains the XML's conservative training default but has not yet been measured
on this corrected pressure-driven model; the older batch 4/8/16 measurements
belonged to rejected flat-axis or STM64 prototypes and do not set current
limits.

The axes have independent truncation telemetry. Over all 7,360 complete
batch-4 corpus rows previously audited, 30 sentences (0.41%) exceed the outer
64-word traversal. Raw word-constituent width is dynamically staged rather
than capped at PS width; the observed corpus maximum was 17, and the regression
trace deliberately passes a 20-constituent word through PS8 without a cut.

The serial path currently contains a host `Tensor.tolist()` boundary that
strict `torch.compile(fullgraph=True)` rejects.  Run eagerly until that boundary
is hoisted out of the captured forward:

```sh
cd basicmodel
BASICMODEL_DEVICE=mps MODEL_COMPILE=none .venv/bin/python bin/train.py \
  --model data/MM_nanochat_grammar_gate.xml --data text --batch-size 1
```

After the behavioral gate passes, launch the comparison tier conservatively
and raise batch size only after observing MPS memory:

```sh
cd basicmodel
BASICMODEL_DEVICE=mps MODEL_COMPILE=none .venv/bin/python bin/train.py \
  --model data/MM_nanochat_grammar_pilot.xml --data text \
  --batch-size 4 --test 20
```

The XML defaults retain resumable, periodic checkpoints. Both tiers still need
batch-size probes before a long run; only gate batch 1 is established for the
corrected controller.

*The whole of `KernelRetirement.md` (190 lines), moved 2026-10-30.*

# Frame-kernel test migration

September 20 frame-kernel retirement audit against BasicModel `08eebe5`.
`test_thinking_kernel.py` had 67 tests. The first landing retained five
TruthInterval tests and the grammar depth campaign; its 61 deletions appear
below. The review removes the unused TruthInterval and its five tests as well;
the grammar campaign remains. Additional dispositions follow this first table.
“mechanics, dead” names behavior intentionally removed with its controller;
it is not a passing behavioral test.

## Closure status contract

The normal controller preserves two explicit support degrees. At the existing
0.5 posture threshold, conflicting support on both sides is `BOTH`. The old
`mixed` categorical closure is dropped: one supported side yields `TRUE` or
`FALSE`, while the weaker opposing degree remains visible. Both weak sides
remain `UNKNOWN`. A budget cutoff with no supported result is `UNKNOWN` plus
`work_budget` incompleteness and a cutoff record; acquired support is not erased
by later exhaustion. The old `bounded_unknown` label and frame closure enum are
not part of the public result. TruthInterval is now removed; no runtime consumer existed. These choices are exercised by the fact-conflict and cutoff probes
below. An early conclude cannot invent a verdict.

Speculative proof execution never materializes a world lemma. Retaining a
completed nested grammatical description is a separate occurrence-write
boundary: embedded content remains unverified or a question, never a fact.

## Every removed test

| Former test | Replacement or disposition |
|---|---|
| `TestLookup.test_stored_idea_is_true` | [test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees](../test/test_replacement_kernel_contracts.py#L81) — Restated on the normal controller. |
| `TestLookup.test_empty_store_is_unknown` | [test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees](../test/test_replacement_kernel_contracts.py#L81) — Restated on the normal controller. |
| `TestLookup.test_no_chaining` | mechanics, dead — the separate direct-only kernel lookup stage is gone; the canonical taxonomy reader traverses a bounded path. |
| `TestLookup.test_direct_taxonomic_edge` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Reader traversal is not evidence of policy-induced depth. The separate `test_policy_selects_nested_what_part_and_receives_actual_episode_credit` review probe covers chooser-selected descent and credit; useful decomposition remains unproven. |
| `TestLookup.test_negative_world_relation_is_not_taxonomic_evidence` | [test_world_refutation_or_conflict_never_closes_taxonomy_false](../test/test_replacement_kernel_contracts.py#L70) — Restated on the normal controller. |
| `TestLookup.test_conflicting_world_relations_are_not_taxonomic_evidence` | [test_world_refutation_or_conflict_never_closes_taxonomy_false](../test/test_replacement_kernel_contracts.py#L70) — Restated on the normal controller. |
| `TestLookup.test_open_binary_query_is_unknown` | [test_open_part_form_executes_a_taxonomy_set_without_an_alias](../test/test_thought_operation_catalog.py#L255) — Changed contract: an open operand returns a typed set, including an empty set, instead of an unsupported scalar lookup. |
| `TestPart.test_up_and_down` | [test_canonical_form_derives_closed_and_open_roles_without_alias_methods](../test/test_thought_operation_catalog.py#L231) — Typed open-role assignment replaces mode/direction strings. |
| `TestPart.test_mode_and_domain_match_the_actual_source` | [test_canonical_form_derives_closed_and_open_roles_without_alias_methods](../test/test_thought_operation_catalog.py#L231) — Typed open-role assignment replaces mode/direction strings. |
| `TestPart.test_bad_mode_or_direction_raises` | mechanics, dead — mode/direction strings are removed; grammar role/domain validation remains in the checked registry tests. |
| `TestCurriculum.test_depth0_lookup_answer` | [test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees](../test/test_replacement_kernel_contracts.py#L81) — Restated on the normal controller. |
| `TestCurriculum.test_depth0_direct_edge` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Reader traversal is not evidence of policy-induced depth. The separate `test_policy_selects_nested_what_part_and_receives_actual_episode_credit` review probe covers chooser-selected descent and credit; useful decomposition remains unproven. |
| `TestCurriculum.test_depth1_syllogism_nested_think` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Reader traversal is not evidence of policy-induced depth. The separate `test_policy_selects_nested_what_part_and_receives_actual_episode_credit` review probe covers chooser-selected descent and credit; useful decomposition remains unproven. |
| `TestCurriculum.test_depth2_two_hops` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Reader traversal is not evidence of policy-induced depth. The separate `test_policy_selects_nested_what_part_and_receives_actual_episode_credit` review probe covers chooser-selected descent and credit; useful decomposition remains unproven. |
| `TestCurriculum.test_world_refutation_does_not_close_taxonomy_false` | [test_world_refutation_or_conflict_never_closes_taxonomy_false](../test/test_replacement_kernel_contracts.py#L70) — Restated on the normal controller. |
| `TestCurriculum.test_dead_end_is_unknown` | [test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm](../test/test_replacement_kernel_contracts.py#L38) — Restated on the normal controller. |
| `TestCurriculum.test_isolated_leaf_unknown` | [test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees](../test/test_replacement_kernel_contracts.py#L81) — Restated on the normal controller. |
| `TestFramesAndBudget.test_budget_exhaustion_bounded_unknown` | [test_budget_exhaustion_is_bounded_unknown_with_diagnostic](../test/test_replacement_kernel_contracts.py#L49) — Restated on the normal controller. |
| `TestFramesAndBudget.test_stack_pops_clean` | [test_nested_cutoff_drains_the_actual_depth_without_fresh_work](../test/test_unified_thought_controller.py#L195) — Restated on the normal controller. |
| `TestFramesAndBudget.test_child_returns_result_not_scratch` | [test_subgoal_carries_the_typed_child_result_into_the_answer](../test/test_thought_answer_adapters.py#L53) — Restated on the normal controller. |
| `TestFramesAndBudget.test_cycle_terminates` | [test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm](../test/test_replacement_kernel_contracts.py#L38) — Restated on the normal controller. |
| `TestFramesAndBudget.test_unknown_op_raises` | [test_structural_operator_without_thought_declaration_is_not_a_boundary_action](../test/test_thought_operation_catalog.py#L92) — Undeclared operations cannot enter the executable menu. |
| `TestInvariants.test_unsupported_assertion_refused` | [test_concluding_without_evidence_cannot_assert_truth](../test/test_replacement_kernel_contracts.py#L59) — Restated on the normal controller. |
| `TestInvariants.test_speculation_does_not_write_ltm` | [test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm](../test/test_replacement_kernel_contracts.py#L38) — Restated on the normal controller. |
| `TestInvariants.test_success_without_materialize_flag_does_not_write` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Reader traversal is not evidence of policy-induced depth. The separate `test_policy_selects_nested_what_part_and_receives_actual_episode_credit` review probe covers chooser-selected descent and credit; useful decomposition remains unproven. |
| `TestInvariants.test_taxonomy_derivation_does_not_materialize_a_world_lemma` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Reader traversal is not evidence of policy-induced depth. The separate `test_policy_selects_nested_what_part_and_receives_actual_episode_credit` review probe covers chooser-selected descent and credit; useful decomposition remains unproven. |
| `TestInvariants.test_direct_hit_does_not_rewrite` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Reader traversal is not evidence of policy-induced depth. The separate `test_policy_selects_nested_what_part_and_receives_actual_episode_credit` review probe covers chooser-selected descent and credit; useful decomposition remains unproven. |
| `TestQueryTestimony.test_registered_addressee_returns_testimony` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_unknown_addressee_zero_trust` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_testimony_is_not_truth` | [test_prediction_and_observation_occurrences_cannot_certify_their_content](../test/test_existence_ingestion.py#L16) — Unaccepted external/predicted content has no fact authority; the testimony API itself is gone. |
| `TestQueryTestimony.test_incorporate_above_floor_moves_lookup` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_incorporate_below_floor_refused` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_incorporate_false_testimony_negative_row` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_legacy_incomplete_relation_cannot_certify_taxonomy` | [test_world_refutation_or_conflict_never_closes_taxonomy_false](../test/test_replacement_kernel_contracts.py#L70) — Restated on the normal controller. |
| `TestQueryTestimony.test_arma_addressee_registered_with_model` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestTestimonyInLoop.test_reliable_oracle_grounds_true` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_refuting_oracle_grounds_false` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_unreliable_testimony_stays_unknown` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_testimony_in_loop_does_not_write_ltm` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_tensor_testimony_is_content_not_truth` | [test_typed_prediction_result_is_detached_before_history_or_checkpoint](../test/test_normal_thought_controller.py#L171) — Prediction remains a typed estimate with no truth support. |
| `TestTestimonyInLoop.test_consult_gate_off_skips_query` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_each_addressee_asked_once` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestRewardsAndTraces.test_success_earns_terminal` | mechanics, dead — verifier-derived terminal bonuses are removed; the one policy uses supplied-answer quality and actual work. |
| `TestRewardsAndTraces.test_step_costs_charged` | [test_policy_charges_shared_episode_work_not_the_number_of_choices](../test/test_unified_thought_controller.py#L132) — Restated on the normal controller. |
| `TestRewardsAndTraces.test_valid_unknown_is_not_failure` | [test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm](../test/test_replacement_kernel_contracts.py#L38) — Unknown is valid; no separate verifier success/failure reward exists. |
| `TestRewardsAndTraces.test_unsupported_assertion_is_failure` | [test_concluding_without_evidence_cannot_assert_truth](../test/test_replacement_kernel_contracts.py#L59) — Unsupported closure cannot assert true; its old terminal-penalty compiler is removed. |
| `TestRewardsAndTraces.test_trace_examples_only_from_grounded` | mechanics, dead — baseline trace cloning and its exporter are deleted; no imitation labels are manufactured. |
| `TestNextOpPolicy.test_featurize_and_logits_shapes` | [test_normal_controller_policy_sees_all_mandatory_roles_and_gets_credit](../test/test_normal_thought_controller.py#L346) — Restated on the normal controller. |
| `TestNextOpPolicy.test_untrained_head_is_neutral` | [test_untrained_chooser_exactly_reproduces_baseline_at_every_level](../test/test_replacement_kernel_contracts.py#L101) — Restated on the normal controller. |
| `TestNextOpPolicy.test_next_op_loss_trains_the_head` | [test_runbatch_credits_each_controller_row_from_its_own_answer](../test/test_unified_thought_controller.py#L147) — Real optimizer membership and a nonzero parameter update from distinguishable row-local policy loss; no claim of held-out utility. |
| `TestNextOpPolicy.test_empty_examples_returns_none` | [test_policy_masks_other_rows_and_drops_failed_episode_graphs](../test/test_unified_thought_controller.py#L215) — No eligible completed records means no policy loss or baseline update. |
| `TestNextOpPolicy.test_head_prefers_stop_short_circuits` | [test_concluding_without_evidence_cannot_assert_truth](../test/test_replacement_kernel_contracts.py#L59) — Restated on the normal controller. |
| `TestNextOpPolicy.test_head_preferring_explore_keeps_baseline` | [test_untrained_chooser_exactly_reproduces_baseline_at_every_level](../test/test_replacement_kernel_contracts.py#L101) — Restated on the normal controller. |
| `TestModelIntegration.test_knob_parsed` | [test_gates_on](../test/test_reasoning_cde_model.py#L33) — Restated on the normal controller. |
| `TestModelIntegration.test_truthset_provisions_rows` | [test_truthset_provisions_source_rows](../test/test_reasoning_cde_model.py#L39) — Restated on the normal controller. |
| `TestModelIntegration.test_off_returns_none` | [test_public_disabled_budget_never_opens_an_episode](../test/test_replacement_kernel_contracts.py#L115) — Restated on the normal controller. |
| `TestModelIntegration.test_think_about_honest_unknown` | [test_reason_about_returns_honest_posture](../test/test_reasoning_cde_model.py#L105) — Unnamed vectors are rejected before a native taxonomy request is formed. |
| `TestModelIntegration.test_syllogism_over_live_taxonomy` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Reader traversal is not evidence of policy-induced depth. The separate `test_policy_selects_nested_what_part_and_receives_actual_episode_credit` review probe covers chooser-selected descent and credit; useful decomposition remains unproven. |
| `TestModelIntegration.test_thinking_loss_head_built_and_in_optimizer` | [test_runbatch_credits_each_controller_row_from_its_own_answer](../test/test_unified_thought_controller.py#L147) — Restated on the normal controller. |
| `TestModelIntegration.test_thinking_policy_loss_graceful` | [test_policy_masks_other_rows_and_drops_failed_episode_graphs](../test/test_unified_thought_controller.py#L215) — Restated on the normal controller. |
| `TestModelIntegration.test_answer_query_kernel_attachment_gates` | [test_public_answer_accepts_the_completed_meaning_without_surface_dispatch](../test/test_unified_thought_controller.py#L36) — One selected summary; no kernel attachment or detector shim. |

## Additional parity-test migration

The full September 20 run also exposed two old presentation-loop tests in
`test_what_spacetime.py`. Their causal-child and bounded-return contracts are
restated by `test_child_context_can_refine_before_returning_its_causal_result`
and `test_nested_cutoff_drains_the_actual_depth_without_fresh_work` in
[test_unified_thought_controller.py](../test/test_unified_thought_controller.py).
The old `test_thinking_uses_completed_subquestion_context_and_restores_parity`
and `test_thinking_limit_forces_every_nested_question_closed` now test the
single-presentation wrapper and rejection of an unfinished presentation without
fabricated closure. Its old repeated-forward and forced parity-close mechanics
are dead. Standalone LTM slot/lifecycle tests remain.

## Nineteen additional deletions in `288b56b`

Audited by qualified test names against `08eebe5`, not by deleted line count.
The descriptions distinguish retired mechanics from surviving contracts.

| Removed test | Disposition |
| --- | --- |
| `test_truth_grounded_reasoning.py::TestLegacyVectorProposalTools.test_generator_surfaces_verified_intermediate` | Retired vector-proposal generator; no replacement claim for learned bridge discovery. |
| `test_truth_grounded_reasoning.py::TestLegacyVectorProposalTools.test_ideas_ranked_by_relevance` | Retired vector-proposal ranking; the ordinary chooser scores grammar requests. |
| `test_truth_grounded_reasoning.py::TestLegacyVectorProposalTools.test_inert_without_spaces_falls_back` | Retired fallback controller; public disabled/unavailable requests are covered by `test_public_disabled_budget_never_opens_an_episode`. |
| `test_truth_grounded_reasoning.py::TestLegacyVectorProposalTools.test_iterations_capped_and_terminates` | `test_budget_exhaustion_is_bounded_unknown_with_diagnostic` on the shared meter. |
| `test_truth_grounded_reasoning.py::TestLegacyVectorProposalTools.test_leaf_istrue_delegates` | `TestEvaluate.test_is_true_true` and ordinary complete-fact probes retain checked fact lookup; facade delegation is retired. |
| `test_truth_grounded_reasoning.py::TestLegacyVectorProposalTools.test_stored_chain_without_generator` | `test_direct_and_depth_one_two_chains_keep_native_premises` retains native chain evidence, without a world-row proposal route. |
| `test_truth_grounded_reasoning.py::TestReasonPredictNext.test_blend_differentiable_masks_absent_arma` | Retired `reason_predict_next` blend/scorer. Typed ARMA prediction remains covered by `test_typed_prediction_result_is_detached_before_history_or_checkpoint`. |
| `test_truth_grounded_reasoning.py::TestReasonPredictNext.test_none_without_generator` | Retired prediction generator; no fallback module is manufactured. |
| `test_what_thinking_episode.py::test_checkpoint_round_trips_the_step_chooser` | `test_normal_controller_checkpoint_rebuilds_its_width_owned_policy`; the obsolete parity schema is discarded. |
| `test_what_thinking_episode.py::test_forced_closure_answers_are_row_shaped_in_a_batch` | Parity closure mechanics retired; normal row isolation/answer ownership covered by `test_runbatch_credits_each_controller_row_from_its_own_answer`. |
| `test_what_thinking_episode.py::test_limit_forces_lifo_closure_with_a_scoreable_root` | `test_nested_cutoff_drains_the_actual_depth_without_fresh_work`; no fabricated parity answers. |
| `test_what_thinking_episode.py::test_rows_reach_parity_independently` | Parity protocol retired; normal row separation covered by the two-row credit and memory-isolation probes. |
| `test_what_thinking_episode.py::test_scripted_episode_opens_executes_and_closes_in_lifo_order` | `test_child_context_can_refine_before_returning_its_causal_result` plus the actual MLP descent/credit probe. |
| `test_what_thinking_episode.py::test_subanswer_is_conditioned_on_the_active_subquestion` | `test_normal_controller_policy_sees_all_mandatory_roles_and_gets_credit` and the causal-child probe; old lexical subquestion wrapper retired. |
| `test_what_thinking_episode.py::test_subquestion_answers_condition_the_root_through_ltm` | `test_child_context_can_refine_before_returning_its_causal_result` checks causal sources; review memory-attention probe checks actual read content. |
| `test_math_thinking_training.py::test_episode_boundary_keeps_slots_live_until_the_step` | `test_episode_end_releases_question_and_trace_credit_together` and actual two-row runBatch credit; ordinary history replaces parity slots. |
| `test_math_thinking_training.py::test_policy_credit_trains_the_step_chooser_and_is_reported` | `test_runbatch_credits_each_controller_row_from_its_own_answer` uses the sole policy and actual optimizer. |
| `test_math_thinking_training.py::test_root_loss_reaches_earlier_iteration_state_under_episode_detach` | Old repeated-forward parity path retired. Live root/active gradient and episode teardown are covered by normal-controller and thought-checkpoint credit tests; no equivalence claimed for deleted parity topology. |
| `test_math_thinking_training.py::test_runbatch_drives_an_episode_and_scores_the_root_after_parity` | `test_runbatch_credits_each_controller_row_from_its_own_answer`; root answer is scored after its ordinary episode, with one optimizer step. |

## Deletions in the corrective review

The rejected codec gate remains **open**. Removing its tests is not counted as
passing grammar learning. The ordinary generation walk retains its existing
word/work-budget tests, typed-answer ownership probes and real optimizer tests.
The fitted nested-choice test establishes selection/credit only, not utility.

| Removed test | Disposition |
| --- | --- |
| `test_chooser_architecture.py::test_default_step_head_preserves_legacy_weights_rng_and_answer_tie` | Replaced by `test_default_step_head_preserves_rng_and_first_candidate_tie` for the sole SelectedThoughtChooser. |
| `test_learned_linguistic_meaning.py::test_meaning_supervision_trains_through_runbatch_after_output` | Rejected fourth interpreter/realiser. Grammar-owned natural-wording gate reopened; no replacement learning result claimed. |
| `test_learned_linguistic_meaning.py::test_natural_parthood_is_learned_then_shared_with_thought_and_generation` | Rejected fourth interpreter/realiser. Grammar-owned natural-wording gate reopened; no replacement learning result claimed. |
| `test_reasoning_cde_model.py::TestReasoningCDEModel.test_answer_loss_actually_trains_the_head` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_reasoning_cde_model.py::TestReasoningCDEModel.test_eager_generator_joins_optimizer` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_reasoning_cde_model.py::TestReasoningCDEModel.test_training_step_runs_the_hook_without_crashing` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_selected_nested_meaning.py::test_learned_lexical_policy_cannot_replace_an_explicit_nested_fold` | Codec precedence mechanic retired. Pure nested selected-fold recovery and three observation writers remain tested in this file. |
| `test_taxonomy_unknown.py::test_empty_legacy_interval_is_unknown_at_zero_threshold` | Unused numerical compatibility value retired. Native support/posture contracts remain in `test_replacement_kernel_contracts.py` and `test_taxonomy_unknown.py`. |
| `test_thinking_kernel.py::TestTruthInterval.test_below_tau_is_unknown` | Unused numerical compatibility value retired. Native support/posture contracts remain in `test_replacement_kernel_contracts.py` and `test_taxonomy_unknown.py`. |
| `test_thinking_kernel.py::TestTruthInterval.test_conflicting_two_sided` | Unused numerical compatibility value retired. Native support/posture contracts remain in `test_replacement_kernel_contracts.py` and `test_taxonomy_unknown.py`. |
| `test_thinking_kernel.py::TestTruthInterval.test_empty_is_unknown` | Unused numerical compatibility value retired. Native support/posture contracts remain in `test_replacement_kernel_contracts.py` and `test_taxonomy_unknown.py`. |
| `test_thinking_kernel.py::TestTruthInterval.test_mixed_luminous_straddle` | Unused numerical compatibility value retired. Native support/posture contracts remain in `test_replacement_kernel_contracts.py` and `test_taxonomy_unknown.py`. |
| `test_thinking_kernel.py::TestTruthInterval.test_one_sided_true_false` | Unused numerical compatibility value retired. Native support/posture contracts remain in `test_replacement_kernel_contracts.py` and `test_taxonomy_unknown.py`. |
| `test_thought_answer_adapters.py::test_generated_set_shares_one_row_word_budget` | Codec-specific budget mechanic retired. Typed set members survive resolve/reverse in the same file; the ordinary generate walk has bounded-output tests in `test_output_walk.py`. |
| `test_truth_grounded_reasoning.py::TestAnswerLoss.test_answer_loss_differentiable` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_truth_grounded_reasoning.py::TestAnswerLoss.test_answer_loss_lower_when_correct` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_truth_grounded_reasoning.py::TestAnswerLoss.test_proof_score_maps_signed_to_unit` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_truth_grounded_reasoning.py::TestAnswerPolicyLoss.test_examples_from_store_builds_transitive_pairs` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_truth_grounded_reasoning.py::TestAnswerPolicyLoss.test_policy_loss_none_without_examples_or_spaces` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_truth_grounded_reasoning.py::TestAnswerPolicyLoss.test_policy_loss_skips_oversized_space` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_truth_grounded_reasoning.py::TestAnswerPolicyLoss.test_policy_loss_trains_generator_head` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_truth_grounded_reasoning.py::TestGrammarOps.test_part` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestGrammarOps.test_parts_is_inverse_of_wholes` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestGrammarOps.test_wholes_proximal_frontier` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyChain.test_beam_caps_candidate_count` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyChain.test_direct_ranks_first` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyChain.test_min_trust_is_weakest_hop` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyChain.test_no_chain_returns_empty` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyChain.test_socrates_syllogism` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyConsolidation.test_chain_to_target_is_shared_loop` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyConsolidation.test_wholes_is_canonical_conceptualspace_method` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyIsPartDirect.test_disjoint_is_none` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyIsPartDirect.test_geometric_containment` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyIsPartDirect.test_negative_trust_row_not_accepted` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyIsPartDirect.test_stored_partof_row` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyMaterialize.test_below_floor_not_written` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyMaterialize.test_materialize_noop_without_store` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyMaterialize.test_verified_chain_becomes_direct_hit` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyPartialOrder.test_materialize_accepts_acyclic` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyPartialOrder.test_materialize_rejects_cycle` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestLegacyPartialOrder.test_materialize_rejects_self_loop` | Geometric/world-row parthood and lemma-writing experiment retired. Native taxonomy contracts remain in `test_taxonomy_view.py`, `test_taxonomy_unknown.py` and `test_replacement_kernel_contracts.py`; numerical containment does not establish taxonomy truth. |
| `test_truth_grounded_reasoning.py::TestSoftGenerator.test_generator_proposes_and_recurs` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |
| `test_truth_grounded_reasoning.py::TestSoftGenerator.test_query_head_is_differentiable` | Separate soft bridge policy, loss and optimizer mechanism retired. Actual sole-controller credit/optimizer behavior is covered by `test_unified_thought_controller.py`. |

`TestSoftGenerator.test_where_read_grounds_in_real_keys` is retained unchanged as
`TestSoftRead.test_where_read_grounds_in_real_keys`; only its class label changes
now that the generator is gone.

*The whole of `BasicModelStabilityPlan.md` (160 lines), moved 2026-10-30.*

# BasicModel numerical-stability plan

Status: proposal only. This document describes the smallest intended path to
make the canonical `BasicModel.xml` run stable on MPS. It is not authorization
to commit the current experimental worktree changes, and it deliberately does
not change legacy models by default.

## What has been observed

The compiled six-hour MPS trial stopped at batch 35 before its optimizer step:
the radix PartSpace codebook gradient contained 4,096 NaN/Inf values. The
failure was caught before weights were written. Replaying from the checkpoint
immediately before that batch still failed after a conservative verb-spectrum
clamp, so that clamp is not a sufficient explanation or repair.

Two independent numerical risks are now known:

1. The old membership butterfly performs power/log-domain mixing close to the
   membership corners. Its forward values can be finite while the derivative
   of a cross term becomes very large at 0 or 1. A De Morgan word union makes
   such near-corner values ordinary, not exceptional.
2. Some concept paths still use a positive `softplus` atom. That is incompatible
   with the desired signed, bounded concept carrier and can create a value
   larger than 1 at the first symbolic/conceptual order.

The first item is the leading hypothesis for the batch-35 gradient failure,
but it is not yet a proven causal trace. The plan therefore begins with an
instrumented reproduction rather than silently converting a bad gradient into
a good one.

## Contracts to preserve

| Carrier | Contract | Notes |
| --- | --- | --- |
| PS/WS WHAT | `[0,1]` membership | `0` is absent, `1` is fully present. WHERE/WHEN remain signed auxiliary coordinates. |
| Base word assembly | `1 - product(1 - m_i)` | Many-to-one; its reverse uses constituent IDs and fold provenance, not a numerical inverse. |
| Unary order raise | invertible `[0,1]^D -> [0,1]^D` sigma/pi | The subject of this stability fix. |
| Concept atom | signed unit row | A bounded activation times that row is coordinatewise in `[-1,1]`. |
| Concept dimension increase | indexed conceptual-codebook read | This is not a sigma dimensional transform. |

No `tanh`/`atanh` carrier chart is required to enforce these contracts. A
straight-through range rail may protect against producer drift, but valid
values must pass through unchanged.

## Proposed canonical-only repair

### 1. Reproduce and localize the failure

Run the saved pre-batch-35 checkpoint against the same shard/order with a
diagnostic mode that records, for each word iteration:

- min/max and finite counts for PS and WS WHAT before and after each fold;
- min/max and finite counts for each fold's input and parameter gradients;
- the first tensor/operation that becomes non-finite.

The diagnostics must stop at the first non-finite value and preserve the
checkpoint. A finite-gradient preflight remains a safety stop before
`optimizer.step`; it is not the proposed cure and must not clip, replace, or
mask gradients.

### 2. Add an opt-in bounded membership fold for BasicModel

Create a new canonical-only sigma/pi implementation, selected explicitly by
`BasicModel.xml`. Do not globally replace the existing meronomy adapter while
this is being validated.

Its butterfly node is a composition of endpoint-safe monotone coupling maps.
For a non-negative bounded coefficient `c`, sigma uses an extensive shear and
curve such as:

```text
E_c(u, v) = u - (1-u) * expm1(-c*v)
h_c(u)    = u + c*u*(1-u)
```

`expm1` matters: the algebraically equivalent `1 - (1-u)*exp(-c*v)` loses
very small memberships to subtraction in float32. Pi is implemented directly
as the complement-conjugate contractive flow, rather than forming `1-x` around
the entire butterfly:

```text
P_c(u, v) = u * exp(-c*(1-v))
k_c(u)    = u - c*u*(1-u)
```

Each scalar curve and shear has a closed-form inverse. Thus sigma/pi preserve
the membership cube and remain invertible for a single unary order raise;
they are not being asked to invert the many-to-one word union.

The trainable coefficients will be smoothly capped and initialized near zero.
For `L` butterfly levels, the cap is budgeted conservatively so the complete
inverse's Jacobian gain is bounded. In particular, the exponential-shear
cross-derivative must be included in this budget; treating `dMaxStable=4` as
only a diagonal cap is not sufficient. Reject non-finite or sub-unit
`dMaxStable` configuration values at construction.

### 3. Keep concept formation bounded without softplus

For the canonical aligned serial path:

1. take each PS/WS native-fold RMS activation in `[0,1]`;
2. perform the indexed lookup of a signed, unit-norm conceptual row;
3. multiply activation and row to produce a conceptual WHAT vector in
   `[-1,1]`;
4. average P0's RMS(union), H0's RMS(property state), the six learned PS/WS
   sources, and an existing prior-STM peer, which
   retains the coordinate bound.

The parallel conceptual decoder must follow the same signed-row rule: remove
the `softplus(dictionary)` atom transform. At initialization and after an
optimizer update, only the touched conceptual rows are reprojected to unit
norm. This is compatible with a million-row codebook because it does not scan
the whole dictionary.

### 4. Keep membership reads on the membership path

The canonical PartSpace radix store and the small WholeSpace property basis
must read `[0,1]` membership rows through a single transformed-read surface.
This includes indexed lookup, full prototype reads, and property
materialization. It prevents an out-of-range master parameter between updates
from re-entering the forward path through a raw read.

## Scope boundary

The initial repair should be limited to the canonical BasicModel path:

- an explicit `BasicModel.xml` opt-in for the bounded membership fold;
- its construction and call sites in the BasicModel PS/WS path;
- the aligned conceptual decoder and its sparse, touched-row projection;
- targeted CPU/MPS tests and the diagnostic runner.

It should **not** initially change the global `SubSpace.normalize` semantics,
the default behaviour of the shared `MeronymicFoldAdapter`, or signed
Lexicon/XOR/legacy configurations. Those use different carriers and require a
separate compatibility decision. Existing uncommitted broad changes should be
split, reduced to this boundary, or discarded before any reviewable commit.

## Validation gates

1. Unit tests: exact 0/1 and adjacent float32 memberships; sigma and pi
   round trips; finite input/parameter gradients; comparable-pair
   monotonicity; and no cross-word-slot mixing.
2. Concept tests: an activation times a signed unit row stays in `[-1,1]`,
   including first-order construction; untouched codebook rows do not change.
3. Compile test: MPS `torch.compile` forward/backward for the unary fold at
   `D=128`. Keep inverse/reconstruction outside the compiled training graph
   until it has its own compiled-MPS buffer-budget test.
4. Reproduction test: resume the pre-batch-35 state and pass batch 35 with no
   non-finite value or gradient. Record loss and sentence rate.
5. Fresh run: at least 60 batches, then a 15-minute run. Start the six-hour
   run only if loss is finite and decreasing and MPS memory is stable.

## Decision points

If the instrumented replay identifies a non-fold source first (for example a
router or loss term), fix that source instead and keep the bounded fold as a
separate experiment. If the canonical-only bounded fold passes the gates, its
behavior and compatibility with older model families can be reviewed as a
second, explicit migration rather than being included in the instability fix.

# Mereology.md

*From `Mereology.md` lines 95-104, moved 2026-10-30.*

> **Codebook IS the meronymic structure.** The standalone
> `MereologicalTree` sidecar that formerly stored explicit parent /
> equality links was retired in favour of pure-geometric parthood
> on the `WholeSpace` bivector codebook. The grammar layers
> `PartLayer`, `IsEqualLayer`, `EqualLayer`, and `QueryLayer` operate directly on
> codebook bivector activations via clipped cosine projection --- no
> separate adjacency table, no `<architecture><mereologicalTreeSize>`
> XML knob (silently ignored if present). Asserted meronymic
> relations are learned by training pulling the codebook geometry
> into the right configuration.

*From `Mereology.md` lines 163-170, moved 2026-10-30.*

**Exact dominance vs. retrieval score.** `part`/`whole`/`equal`/`overlap`/
`underlap`/`boundary`/`copart` above are the graded clipped-cosine *score*
(see the Meronomy reconciliation note above, item 1). The boolean dominance
trio `Ops.partOf(S1, S2)` / `Ops.wholeOf(S1, S2)` / `Ops.overlapOf(S1, S2)`
(`bin/Layers.py`) is the separate exact ground truth: elementwise $\le$ /
$\ge$ reduced over the last dim with `all` (`overlapOf` is the zero-directed
elementwise min, `Ops._radmin`) --- not a `part`-suite formula, and not what
this page's scores approximate at retrieval time.

*From `Mereology.md` lines 227-300, moved 2026-10-30.*

## ImpenetrableLayer: Five-Relations Regularizer

`ImpenetrableLayer` regularizes the WholeSpace whole-percept codebook. It
classifies each ordered pair of codebook rows $(i, j)$ into one of the five
mereological relations (using `Basis.part`), and penalizes partial overlap
when paired with a trust mismatch. The learned half of the [Codebook
Uniqueness Contract](Spaces.md#codebook-uniqueness-contract).

### Penalty

$$
\mathcal{L} = \mathrm{overlap\_weight} \cdot
   \frac{\sum_{i \neq j} \mathrm{overlap}(i, j)
                        \cdot |\mathrm{trust}(i) - \mathrm{trust}(j)|}
        {K(K-1)}
   + \mathrm{variance\_floor\ term}
$$

Overlap strength is damped to zero as the pair approaches `equal`:

$$
\mathrm{overlap\_strength}(i, j) =
   \min(P[i,j],\ P[j,i]) \cdot
   \left( 1 - \max(P[i,j],\ P[j,i])^k \right)
$$

with $k = \mathrm{equal\_suppression}$ (default 4.0). This is a continuous
formula, not the discrete five-way classification below it: $\min(P[i,j],
P[j,i])$ already reads near-zero for `disjoint` and strict `part` (one or
both scores near 0), so only `equal` needs an explicit damp --- the
$(1 - \max(\dots)^k)$ factor pushes the penalty toward zero as a pair's
mutual parthood approaches 1.

A separate `variance_floor` term guards against row collapse.

### Trust

Per-row usage frequency, derived from VQ EMA counts:

$$
\mathrm{trust}(i) = \frac{\mathrm{cluster\_size}[i]}{\sum_j \mathrm{cluster\_size}[j]}
$$

When VQ is absent, trust falls back to $\|cb[i]\| / \max_j \|cb[j]\|$.

> **Relative relations and tetralemma trust.** A relative sentence in
> the `part` / `isEqual` predicate family is preserved at depth 3 as
> `[predicate, idea1, idea2]` and may be learned into the codebook as a
> ternary META edge (predicate as parent, the two ideas as children) via
> `WholeSpace.insert_relation`. An accepted relation carries a
> **tetralemma trust 4-tuple** $(t, f, b, n)$ (TRUE / FALSE / BOTH /
> NEITHER, summing to $1$) from the TruthSet posture, gated by the
> content-aware learn-score against `<truthCriterion>`. See
> [STM.md Section 9](STM.md#9-relative-vs-absolute-end-states).

### Diagnostics

After `forward()`: `last_overlap_loss` and `last_variance` populate whenever
their weight is nonzero. `last_relation_counts` (dict summing to $K(K-1)$)
populates only under `MODEL_DEBUG` --- the five `.sum().item()` calls it
needs are host syncs that would otherwise break CUDA-graph capture.

### Configuration

XML knobs (under WholeSpace):

| Knob | Default | Meaning |
|------|---------|---------|
| `impenetrableOverlap` | 0.0 | Weight of overlap $\times$ trust-diff penalty |
| `impenetrableVariance` | 0.0 | Minimum variance regularizer weight |
| `impenetrableAntisymmetry` | 0.0 | Legacy antisymmetry weight, still accepted by schema |
| `impenetrableTransitivity` | 0.0 | Legacy transitivity weight, still accepted by schema |

---

*From `Mereology.md` lines 306-312, moved 2026-10-30.*

> **Terminology note.** The cross-tower link that ties one part-percept to one
> whole-percept by reference is a **concept** (a `ConceptualSpace` relation);
> `insert_meta` allocates it into the **Concept codebook** --- the
> `WholeSpace.taxonomy` / `taxonomy_parent_map` / `meta_pair_to_idx` /
> `meta_trust` / `part_chain` tables. The earlier `_sym_*` placeholder name
> no longer exists in code (it survives only in older comments); prose below
> uses "concept" for the relation.

*From `Mereology.md` lines 344-371, moved 2026-10-30.*

Parts are synthesized automatically (the join over atoms, every pass);
wholes are analysed automatically too, by division, and the mechanism is
the Linde-Buzo-Gray split already on WholeSpace (`bin/Spaces.py`,
`record_lbg_pull` / `maybe_split_lbg`, 2026-05-28). As implemented:

- **Pull accumulation.** Every time a percept binds to a WholeSpace row
  (the autobind path in `ConceptualSpace`), the row records the pull the
  percept's code exerts on it: per row, the displacement sum
  `Σ (v − W[row])`, the sum of squares `Σ (v − W[row])²` and the count.
  Cost O(D) per binding; only WS-side rows (raw wholes and META rows
  whose vectors live on the WS codebook) participate; non-finite pulls
  raise (fail loud).
- **Trigger.** A row splits when its count is at least `lbgMinCount`
  (default 8) and the maximum per-coordinate assignment variance,
  `E[x²] − E[x]²` over the accumulated displacements, exceeds
  `lbgThreshold` (default 0.5): the positions bound to the row have
  come to lie in (at least) two places.
- **Direction and split.** The split direction is the unit-norm mean
  displacement, the principal pull (no eigendecomposition). The
  original row moves to `old + ε·d` in place; a fresh row is allocated
  at `old − ε·d` through `insert_whole` (`lbgEpsilon`, default 0.1). The
  split-off row inherits the original's fold provenance, and if the
  original was the child of a META node a META edge is registered for
  the new row too, so both halves stay reachable on the reverse walk.
  The accumulators of the original row reset.
- **Unsupervised.** Nothing about the trigger reads a loss: the
  variance is a property of where the bound codes lie. Training only
  shapes the code geometry over time; the division itself is automatic.

*From `Mereology.md` lines 412-506, moved 2026-10-30.*

### The corpus callosum links the towers (part `isa` whole)

The full mechanism: the **corpus callosum** in ConceptualSpace (the
learned `[2N,N]` `self.callosum` glue) is what joins the towers into one meronomy. A
part-percept `A` (PartSpace) and a whole-percept `B` (WholeSpace) have `.what`
codes from *different* codebooks (incomparable), but their **`.where` is
comparable** — so the callosum asks whether `A.where` is *part of* `B.where`
(a direct span-tuple containment test, `record_cross_tower_meronomy`; see the
2026-07-09 delta below for why this is no longer a `WhereEncoding` decode);
when it is (and no greater part / lesser whole intervenes, per codebook
activation) it forms the **concept `A isa B`** (token `isa` type; the type *names*
the token). Co-occurrence at the same `.where`/`.when` drives the link, weakening
when they dissociate — a **Hebbian** coupling between codebooks.

Since the 2026-06-16 redesign `.where` carried an **endpoint-sum bracket**
`[start, end]` (angle = span center, magnitude = extent); containment was
`A.start >= B.start && A.end <= B.end`, and **contiguity** (adjacency / gap)
between sibling parts was the same endpoint comparison. An instant snaps to
zero extent, so atoms compare as points. (See `doc/Spaces.md` for the
encoding. 2026-07-04: `.when` left the bracket — it is the 4-dim start
ladder; temporal extents ride the record store / exact clock, not the band.
**2026-07-09:** the muxed `.where` band itself moved to a 2-rung quadrature
*ladder* over the byte START only (`WhereEncoding`, `bin/Spaces.py`) — range
rung + resolution rung, no END in the band. `WhereEncoding.decode_span` is
retired. The standalone analyzer's distinct `EndpointSumWhere` codec was
also retired in the October 1 suite trim; the live WhereEncoding tests remain. The
runtime containment test above no longer goes through a `WhereEncoding`
decode at all: `WholeSpace.record_cross_tower_meronomy` (`bin/Spaces.py`) and
`RunStructureLayer.contained_mask` (`bin/Layers.py`) read `.where` span
tuples directly.)

**Design decision (2026-06-29, REVISIT) -- the `.where` support of percepts vs
symbols.** Every PERCEPT (a part from PartSpace or a whole from WholeSpace) MUST
carry a real `.where` (`nWhere > 0`): a definite `[start, end]` extent. A percept
with no location is not admissible. A SYMBOL (a concept-level relation entry, or an
SS reference) MAY instead either (a) carry `nWhere = 0` (no location of its own), or
(b) carry a RECOMBINED `.where` equal to the bounding span of its constituents --
`start` = the leftmost constituent's `start`, `end` = the rightmost constituent's
`end` (the support of a symbol is the sum of the supports of its members). If the
recombination is hard to implement, the fallback is to give every symbol a `.where`
of "everywhere" (universal support). THE ANT-COLONY QUESTION: is the location of an
ant colony IN each ant (distributed -- every ant keeps its own `.where`), or is
there a SINGLE location that subsumes all the ants' locations (the bounding region)?
The recombined bounding-span (b) takes the single-subsuming-extent view; "everywhere"
is its degenerate limit. This choice determines whether a concept's `.where` is
distributed over its parts or collapsed to one subsuming extent, and should be
revisited.

Word $\leftrightarrow$ object cannot be linked this way (too unlike; no convex set is specific
enough), so a **second-order meta-object** is synthesized in PartSpace, **outside
`.where`/`.when`**, fusing the word-code and object-code into the symbol used in
serial communication (the MetaSymbol). *(Language update 2026-07-02, revised
same day by the two-phase rework: in the ramsified CS this is a FIRST-order
META-concept -- the sec-4c ORDERED PAIR $[whole=word\text{-}symbol,
part=object\text{-}symbol]$, roles as positional slots of an ordered pair
rather than containment claims; the typed read-out recovers (word, object) by
INTERSECTION with the word-symbol class (`meta_word_object`), and the
reference-table pairing remains the serial-mode access path. See
doc/Architecture.md sec A.)* Because symbols are outliers, the
part$\leftrightarrow$whole **concepts** live in a **two-code LUT** and the symbolic taxonomy is
**relations over symbol indices** — which are the TruthLayer's **absolute truths**
(propositions) and **relative truths** (`RelativeTruthStore`), to be integrated.
Taxonomy relations are also learned **explicitly from trusted language** ("cats are
furry" $\to$ `[cats] <= [furry]`).

**Granularity / the part-whole ratio** is the correctness signal (and the MM_20M
mean-collapse fix): *many parts $\to$ one whole* = under-analysed; *one part $\to$ one whole*
= over-analysed. An incorrect ratio is the criterion to request **further synthesis
($\sigma$, e.g. radix) or analysis ($\pi$, property tiling) within the problematic `.where`** —
until words emerge as parts. MM_20M collapses because its byte chunker never climbs
and its analyser never descends, so no word-granularity parts exist for XOR.

**Design decision (2026-06-29, REVISIT; brake updated 2026-07-03) -- locking in the
"basic level" of analysis.** The mereological analysis should converge on and
STABILISE at Eleanor Rosch's BASIC LEVEL -- the granularity where the part-whole
ratio is correct (neither many-parts-one-whole nor one-part-one-whole). The radix
climb builds parts UPWARD (byte -> prefix -> word -> ...); left unbounded it
over-climbs (a whole short sentence promotes to one percept). Two principled brakes
hold it at the basic level: (a) a STRUCTURAL bound -- word tiling: a chunk cannot
cross a word boundary, where the cut is WHITESPACE AND PUNCTUATION (separator runs
-- whitespace or punctuation -- are themselves maximal same-class tiles that get
observed/promoted just like word tiles; see ``PartSpace._embed_radix`` under
``_meronomy_words`` and ``RadixLayer``'s ``word_bounded`` promotion gate); and (b)
the part-whole-ratio convergence signal above (request more synthesis/analysis
until the ratio is right). The ideal is to LOCK IN whatever basic level the corpus
implies. As a pragmatic HEDGE -- and because **word-level is essentially the basic
level for a symbolic LLM** -- the analyser cuts on the word tiling (PS
observes/promotes per tile so parts top out at word size), which is a hard
structural bound rather than a size threshold: the earlier ``<basicLevelMinSize>``
/ ``<basicLevelMaxSize>`` letter-count bounds were too loose to separate a short
sentence from its words on their own, could not express "don't cross a boundary"
at all, and are now REMOVED (dead config, never wired to anything -- the word
tiling supersedes them); the full ratio-driven basic-level convergence remains the
to-revisit principled version.


*From `Mereology.md` lines 517-524, moved 2026-10-30.*

> **Terminology note.** This section formerly called the explicit part$\leftrightarrow$whole
> relation entry a "symbol." Per the convention, that relation — one part-percept
> tied to one whole-percept by reference, held in the Concept codebook's
> `taxonomy` / `taxonomy_parent_map` / `meta_pair_to_idx` / `meta_trust` /
> `part_chain` tables — is a **concept**. The 0-D `SymbolSpace` **symbol** is a
> separate, intensional reference *to* such a concept and is not what those
> entries are. Prose below says "concept"; the code no longer carries a
> `_sym_*` placeholder (that name survives only in older comments).

*From `Mereology.md` lines 586-590, moved 2026-10-30.*

- **Overlap penalty (not antisymmetry).** Legacy `ImpenetrableLayer`
  penalized mutual parthood $P[i,j] \cdot P[j,i]$ directly, pushing
  *every* overlapping pair apart including synonyms. The five-relations
  design leaves `equal` pairs alone and only penalizes the ambiguous middle
  region, gated by trust mismatch.

# Orders.md

*From `Orders.md` lines 201-214, moved 2026-10-30.*

- **Ramsify over conceptual spaces.** A concept at order *n* is defined by
  its relations to the symbols of order *n−1* — a kind by which symbols
  are its members, an individual by which occurrences are its, with the
  truth lanes as degree (§3). Those relations are its coordinates: order
  *n*'s conceptual space is the cube over the order-(*n−1*) symbol index,
  a concept there is its characteristic vector, and similarity at order
  *n* is overlap of relation sets — the meet of two sparse membership
  vectors. This is §2 made geometric: intension at order 0, extension
  above. It is never meaningless: a discontinuous type is a sparse vector
  over disjoint members. Order 0 is the base case, its concepts placed by
  context in a dense similarity space — a compression of the same kind of
  relation set, the symbol's contexts ([FutureWork, placement](FutureWork.md#placement-from-the-registered-priming-surface-one-metric-for-sbow-diffusion-and-bind-alec-2026-10-20-proposed-with-the-objective-to-decide)).
  Nothing is stored for the orders above 0 beyond LTM2's `Part(member,
  whole)` edges: the edges *are* the vectors.

*From `Orders.md` lines 248-266, moved 2026-10-30.*

- **The priming surface is the same object, read stratified.** It is one
  vector of energy over the codebook rows with each row's order known
  (`priming_row_orders`). Restricted to the symbols of order *n−1* it is a
  fuzzy set of them — a point in order *n*'s conceptual space: the
  activation over words is the current context as a point in order 1's
  space, over individuals the current scene in order 2's. One vector, one
  point in every order's space at once. **The diffusion kernel per
  stratum:** across orders, along the Part edges (member to whole, whole
  to its other members — the existing two-hop); within an order above 0,
  the same two-hop through shared members, so no code-derived channel
  exists there; within order 0 only, the geometric overlap in the dense
  similarity space — the one code-derived channel, which placement must
  not read. **Read points per consumer:** placement reads the order-0
  stratum after the seen bumps and the edge diffusion, before the
  order-0 geometric channel; `bind` reads the request's order as its
  expectation term and the Part edges for the verb's participants;
  attention and the chooser read all strata; thought's desire channel
  bumps the stratum it names. Decay, bumps, energy conservation and the
  capped read are unchanged ([Architecture, priming](Architecture.md)).

*From `Orders.md` lines 406-423, moved 2026-10-30.*

**The surface places the concepts (Alec, 2026-10-22).** "We are doing
explicit spreading over the priming surface, and then using the activation
of the priming surface to distribute vectors in conceptual space, including
the order-zero concepts. That said, the priming is across orders, and the
concepts each have their own space, so there is still some disconnect."
Agreed, with §9's resolution of the disconnect (reversed in the next
paragraph): the spread crosses orders
along the Part edges and the shared occurrences, but the read is per
stratum. Restricted to the symbols of order n-1, the surface is a point in
order n's space, so each concept is placed by the one surface read at its
own stratum. One energy, many strata, no shared space. In the code the
spread crosses orders (`Priming.propagate` over
`similarity_codebook.mereology.placement`; the `Priming.diffuse` and
`priming_row_orders` names are gone, audit 2026-10-30) and each row's order is
known through the allocator's `order_of`, but no reader yet restricts the surface to a
stratum; that read, with placement, was not yet built. Placement itself is
unchanged: the same meet-raising objective at every order, over the stratum
the concept's order names.

# Logic.md

*From `Logic.md` lines 98-115, moved 2026-10-30.*

## 2. Symbolization

> **Historical (design note only).** The norm-projection formula below has
> no implementation: `Basis.symbolize` was removed in the 2026-05-01
> cleanup (no live callers), and live symbol emission is the
> WholeSpace codebook snap (the quantize/lookup that names a
> whole-percept's prototype row). The formula is kept as the original
> design reading of "symbolization = norm projection".

For $X \in (B, N, D)$:

$$
s(X) = 2 \cdot \mathrm{mean}(\lVert x_i \rVert) - 1 \quad \in [-1, 1]
$$

Interpretation: $+1$ = strong presence, $0$ = neutral, $-1$ = absence.

---

# Philosophy.md

*From `Philosophy.md` lines 557-582, moved 2026-10-30.*

### The rule

With `o` the idea composed from a sentence, $\hat{e}$ the idea that was expected,
$\kappa$ how strongly each role was expected to be filled, `g` a gain and `m` the
attention on each role, what is **conceived** is

```text
$c = o - g \cdot (1 - m) \odot \kappa \odot \hat{e}$
```

A sentence that was perfectly predicted conceives nothing. One that departs
from its prediction conceives the departure. A role that was expected and did
not come conceives $-\hat{e}$: an absence. And $o = c + g(1 - m)\kappa\hat{e}$ exactly while
the estimate is kept, so nothing of the input is lost: the sentence is met
from the expected point instead of from zero. It is a change of origin, not a
distortion.

This is the opposite sign to the natural first idea, that expecting something
should *raise* it in the mind. Raising the expected makes comprehension lean
toward its own prediction: the input is no longer accepted as it is, and a
predictor trained on what was understood is then trained on its own echo.
Subtracting leaves the understanding alone. "If we do not predict, we process
exactly what is there" (Alec); under this rule that is equally true when we
do predict, and the difference expectation makes is confined to what is
*new*.


# ExistenceEvidence.md

*From `ExistenceEvidence.md` lines 142-171, moved 2026-10-30.*

## Historical validation (September 16)

The repository's 51 new probes pass in 3.11 s; ten affected existing files
pass 281 tests with one skip and three warnings in 89.56 s. The full suite
passes with exit status zero: **4279 passed, 51 skipped, 7 xfailed, 185 warnings, 4 subtests passed in 4175.57s (1:09:35)**.

Original lookup, metadata and ingestion probes failed before implementation.
Follow-up candidate probes exposed lost final-result provenance, incomplete
checkpoints, replaced scope, public-input flattening and incomplete legacy
fact admission; each failed before its corresponding fix.

The isolated candidate checks are preparation evidence. Repository checks
above describe the September 16 runtime. Grammatical routing, ordinary thought
history and nested retention now use the one controller. Residual policy credit
remains open. Taxonomy evidence is documented separately
in [Taxonomy queries](TaxonomyQueries.md).

# TaxonomyQueries.md

*From `TaxonomyQueries.md` lines 109-141, moved 2026-10-30.*

## Historical validation (September 16)

The repository's **37 new probes pass in 3.33 s**. The sixteen affected
files pass **277 tests, with one skip, two expected failures and seven
warnings, in 173.39 s**. The full suite passes with exit status zero: **4317 passed, 51 skipped, 7 xfailed, 184 warnings, 4 subtests passed in 5604.66s (1:33:24)**.

Full-suite command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  .venv/bin/python -m pytest test -q -x -p no:cacheprovider
```


The existing public-routing probes failed before the fix. Further failures
caught native read accounting, public entry points, operation-head curriculum,
unrelated child testimony, public-helper bypasses and zero-threshold unknown
classification. Each received a regression test.

# WhatSpacetimeDesign.md

*From `WhatSpacetimeDesign.md` lines 466-502, moved 2026-10-30.*

For each protected perceptual/conceptual parameter tensor, let `r` and `o` be
the reconstruction and output gradients **after** their actual loss weights
and truth modulation. The accepted policy is asymmetric:

```text
if dot(r, o) < 0 and ||r|| > 0:
    q = o - dot(r, o) / dot(r, r) * r
else:
    q = o

q = q * min(1, rho * ||r|| / ||q||)  # zero q stays zero
if r is missing or zero: q = 0
shared_gradient = r + q + separately named auxiliary gradients
0 <= rho < 1                       # initial rho = 0.5
```

Remove only the conflicting projection, preserving useful aligned output
credit. This gives `dot(r, q) >= 0` and `||q|| <= rho * ||r||` within numerical
precision. Reconstruction itself is not projected or downscaled. Independent
symbolic/output heads receive ordinary output credit. Parameters tied into
both analysis and synthesis receive protection on their complete derivative.
Sparse codebooks must stay sparse over touched rows.

This guarantees non-opposing output credit at the gradient level. It does
not guarantee a decrease in reconstruction loss after a finite Adam/momentum
step, nor protect against arbitrary auxiliary objectives; monitor actual
reconstruction quality as well. Report branch norms, alignment, and the
removed/capped contribution when joint output training is active.

The zero-reference case is consequential: an exact inverse identity, detached
reconstruction, or a disconnected encoder gives output no budget on protected
parameters under this policy. Independent output heads can still learn. The
current BasicModel uses `detachedReverse=true` and a non-grad, context-rotated
concept dictionary; enabling projection does not create a reconstruction
gradient through either boundary. Completing the field/answer architecture
must resolve those ownership and credit paths explicitly. No residual channel
is introduced merely to manufacture a reconstruction error.

*From `WhatSpacetimeDesign.md` lines 604-639, moved 2026-10-30.*

## 8. Context for the grammar chooser

> **Target update, 2026-09-12:**
> Sentence-boundary thinking
> supersedes the shared-chooser decision below with separate sentence and
> thought/query controllers. It defines their phase boundary and common
> semantic interface for linguistic subgoals and typed tools. It also replaces
> Q/A pairing and parity with depth-labelled thought contexts and implicit
> serial LIFO return. Preserve append-only history, target isolation and
> reconstruction/output separation; the Q/A layout here describes the earlier
> runtime, not the replacement target.

The grammar chooser already operates over a richer context than a new
address-specific list would describe. Its context includes symbolic
activation, STM, LTM, percepts, and grammatical operations, among the other
state already supplied by the model.

This design does not replace or duplicate that context. It requires the
chooser to be able to distinguish only the new, relevant state:

- the temporal meaning contained in the active `what()` question;
- the input and output sides of LTM slots;
- whether unmatched input-only slots remain; and
- the current closure pressure while thinking.

The chooser may use that context to answer, to leave an input unanswered
while posing a subquestion, or to emit an output for the latest pending input.
Those are grammatical/model choices within the existing run path, not calls
to a separate query planner.

The chooser also controls output realization after the answer is resolved. Its
context includes the answer symbol, current derivation, output sentence
location, already constructed output, and the perceptual/conceptual references
available for synthesis. These values let it choose the next grammatical
derivation and where that derivation contributes to the answer surface.


# Testing.md

*From `Testing.md` lines 535-925, moved 2026-10-30.*

### Fast aggregate-runner evidence (September 19)

The reviewer probes were intentionally red: the old runner had neither an
eight-GiB-reserved default budget nor a parallel-worker API. After the pool
implementation, the aggregate-budget/real-overlap probe passed.
It proves two isolated workers overlap while their
independent caps remain subject to one aggregate reservation.

The complete resource, recycling and device selection then passed **28/28**.
A first 80%-recycling attempt correctly failed its
128 MiB allocator fixture before the hard cap; the fixed 64 MiB headroom made
that same fixture recycle safely, while the time probe verified the new 80%
deadline boundary. The MPS-routing file passed **7/7**, including the
two-worker probe that confirms two
accelerator training workers do not overlap.

### Complete diagnostic coverage (September 19)

The reviewer probe `test_test_failure_does_not_cancel_remaining_worker_coverage`
was red under the old policy: a deliberate first
assertion failure left the inner run at its first selected case. The repaired
policy passed the focused probe: it retains exit
status `1` and `test_failure` for the deliberate red case while completing the
second and third selected cases. Resource, deadline, protocol and coverage
failures remain fail-fast; only ordinary pytest failures receive complete
diagnostic coverage. The complete bounded-runner file then passed **25/25**.

### Selected direct-unary meaning (September 19)

The reviewer probe for a completed `what(quantize(x))` action program was red:
the former binary-only adapter returned no
meaning. The direct concept-unary adapter passed its recovery/gradient probe;
its normal-controller `code` result probe passed
alongside it. The companion arbitrary-ID guard,
which proves that a direct leaf cannot manufacture the description occurrence
needed by `arma`, passed. The source-matched full
default run then passed **4,664/4,664** in **195.2 s**, peaking at
**18.7 GiB** aggregate.

Claude's throughput review then supplied five concrete follow-ups. Their new
reviewer probes were red; the repaired policy passed
5/5: no `-b` throttle, `nice -n 10`, `cpu−4`
execution capacity, full-run-only lock, recorded-but-noninvalidating prose,
and direct xdist iteration. The aggregate-memory kill probe passed: two
workers below their individual caps were stopped
when their combined footprint crossed 256 MiB. These focused runs establish
the resource and routing mechanism only; the current source still requires a
full default run and measured end-to-end throughput before the gate closes.

After those policy changes, the full runner/recycling/device selection passed
**35/35**. It includes the restored direct-xdist
iteration target, lock/snapshot behavior, real child cleanup and recycling,
aggregate cap, ordinary CPU dispatch, explicit MPS routing, and the one-lane
accelerator concurrency probe.

The current fast **default** run passed **4,649/4,649** selected cases once:
31 fresh workers, 10 execution slots,
8 GiB per-worker caps, a 20.20 GiB observed aggregate peak, and 196.80 seconds
end-to-end. The prior serial run took 4,164.65 seconds for 4,640 cases, so
this is a measured current throughput result, not an extrapolation from a
microbenchmark.

**September 18 default-suite coverage record.** The September 17 checkpoint
attempt remains an incomplete historical run: it exited 124 after 5,136
seconds with 2,907 of 4,458 selected cases completed. After the OS update,
Alec directed the runner to finish every case without re-running cases that had
already passed. The resulting composite record covers all **4,502**
node IDs in the then-current default selection exactly once after deduplication:
the last remaining selection passed 273/273 in 386.55 seconds.

Two real failures found while completing that record were fixed and their
specific nodes were re-run green: the output-only walk (30/30), and the peer
language pipeline (23/23). The latter revealed that a local
`LanguageSpace` without an output loop must not construct legacy output-rule
keys.

This is deliberately a **no-rerun composite coverage gate**, not one fresh
single-snapshot success run: source marker changes and the VQ tile-bound
fix occurred while the historical coverage was being completed. It is the
validation form explicitly requested for this session; future source changes
require their ordinary affected tests and a fresh bounded full run.

### Thought-operator catalogue record (September 18)

The catalog-unification snapshot was run through the bounded default selection:
all **4,623** selected node IDs
completed once in 38 serial fresh workers. Exactly one node failed:
`test_signal_router_has_grammar_ops_attached` still assumed that unary grammar
kernels were unwrapped, while the common structural-face contract now wraps
both unary and binary kernels. The production code was not changed for that
failure; the reviewer assertion was corrected to inspect the wrapped layer and
its exact bounded re-run passed **1/1** (19.8 seconds, 0.41 GiB peak).
The explicit guard that keeps deferred `true` out of the production catalogue
also passed **1/1**.

The earlier catalog/contract regression selection passed **137/137**;
direct context/catalog probes passed
**38/38**. Per Alec's explicit
no-rerun instruction, the 4,622 nonfailing default nodes were not rerun after
the assertion-only repair. This is current composite coverage evidence, not a
fresh single-snapshot all-green run; no source behavior changed after the
full selection.

The audit retained every assertion. Full server construction, full traversal
and compiled `runBatch` integration checks are marked `slow`, while their
compact contract checks remain default. A real VQ failure showed that the
former 4 GiB default distance-tile budget could exceed the 8 GiB bounded-worker
cap through allocator retention. The default is now 512 MiB. Its two
large-flat OOM probes passed under the explicit slow CPU gate (2/2; about 29.4
seconds and 1.41 GiB peak per case).

The explicit current supervised MPS training check also passed:
requested and resolved device were MPS, the worker
took 170.806 seconds, and peak aggregate footprint was 2.36 GiB. This proves
routing and bounded execution, not a controlled MPS speedup or a held-out
utility result.

### Explicit `<thought>` catalogue (September 19)

The corrected catalogue began with red bounded reviewer probes:
a `<thought>` selection was ignored, structural-only
forms leaked into the registry, and a conflicting role declaration was
accepted. A selected thought-only form then correctly exposed the missing
structural-family assumption, and an exact
`isPart` declaration exposed its missing checked descriptor.
The repaired loader retains thought declarations
separately, allows selected thought-only forms, joins matching structural
faces only after role validation, and fails closed when no executor exists.

The final affected selections passed **114/114** and the broader
grammar/controller selection passed **311/311**. They cover explicit membership,
structural omission, thought-only forms, exact `isPart`, descriptor failure,
legacy `<Queries>` rejection, phase gates, query VP boundaries and lifecycle
contracts.

The first current default run collected 4,655 nodes and completed
1,727 before one warm 194-case worker jumped to 12.96 GiB and was killed. Its
aggregate peak was only 13.97 GiB, below the 28 GiB suite reservation; this was
worker accumulation, not an aggregate-cap breach or a test assertion failure.
Per the explicit no-rerun instruction, the remaining node IDs were recovered
only once in fresh 16/32-case batches. Green runs covered 1/1, 700/700,
225/225, 1,000/1,000, 2/2, and 46/46. A transient source-snapshot invalidation
occurred after 751 otherwise passing nodes. The stateful
`test_what_thinking_episode.py` module was consequently checked once in its
native collection order (**10/10**) after an
alphabetically reconstructed shard demonstrated that its module fixture cannot
be meaningfully split by node ID.

Together these runs exercise every one of the 4,655 collected default
nodes on the unchanged source, with the corrected thought catalogue tests
green. This is deliberately composite coverage, not a single fresh green
default run: future source changes require their own affected tests and a
new frozen full run.

### Direct lexical meaning and normal-route quarantine (September 19)

The reviewer probe for an unreduced lexical `[NP1, VP, NP2]` relation was red
(**2/2 expected failures**): the
adapter returned no meaning when the middle word was an installed native VP.
The repaired direct adapter passed **2/2**,
preserving the live signed noun leaves,
the canonical native VP, and the intended operand-only gradient route.

The normal-route reviewer probe was red (**1/1 expected failure**) because an
unselected surface prompt still called `answer_query()`. After the production
handoff was removed, its exact rerun passed **1/1**; the dense compatibility
topology was also protected by **2/2**.
The combined meaning/catalogue/controller/phase/work selection passed
**116/116**. This verifies the
narrow adapter and normal-path quarantine; it does not claim completion of
paraphrase, nested-reference, residual-credit, learned-utility, or full-suite
gates.

The affected dense output-synthesis module also passed **26/26**.
The first current default run then
collected 4,658 nodes and passed 2,812 before the old
`TestWriteMask.test_partition_isolation` smoke test exceeded its worker cap
after twelve earlier legacy-model constructions in that same process
(11.23 GiB). The still-uncompleted node passed alone
**1/1** at 1.76 GiB, proving cumulative worker
state rather than an intrinsically oversized test. The exact remaining 1,845
node IDs then passed in fresh 8-case workers
(302.13 seconds; 12.24 GiB aggregate peak).

The three runs together cover
each current default node exactly once: **2,812 + 1 + 1,845 = 4,658** passing
cases. This is current-source composite default coverage under the explicit
no-rerun policy, not a single green run.

### Anchored lexical converse provenance (September 19)

The reviewer probe for an unreduced lexical `whole` converse was red
(**1/1 expected failure**): recovery
selected the first shared-VP form (`part`) and reversed the canonical roles.
The repair passed **8/8** focused selected-meaning/native-ID cases.
It retains a detached grammar-form
classification rather than a raw surface, row, or native-ID feature.

A second owner probe was red
(**1/1 expected failure**): it proved that form classification had incorrectly
read the raw InputSpace record rather than the PartSpace word segmentation.
That exact probe and the anchor-spelling/converse check passed **2/2**;
the complete affected thought,
registry, phase, and selected-meaning set then passed **135/135**.

The first fresh default attempt exposed a clone-safety regression from using a
non-pickleable `MappingProxyType` for the private grammar snapshot:
it stopped at **1,654/4,660**, with the two
deep-copy tests in `test_compiled_word_chunk.py` failing. The two failures
were reproduced red, then the
clone-safe copied snapshot plus lexical probes passed **4/4**.

In the new frozen full default run, **4,660/4,660** cases passed in 195.49
seconds, at a 18.74 GiB aggregate peak under the 28 GiB cap.

### Forward-owned lexical forms (September 19)

The review of `943a64f` identified a second text lookup at program capture:
`_word_lexical_forms` reclassified PartSpace's transient `word_texts` through
the anchor table instead of retaining the forward's decision. The row-dispatch,
missing-provenance, and ordinary-forward probes were red.
That run also caught a packed-fixture
bucket mismatch; after using a fixed 16-word fixture, both ordinary and packed
forward probes failed for the intended missing-form reason.

The eager word-row staging now resolves each WORD's form once and retains it
by WORD row. Capture neither reads text nor repeats the anchor lookup, and
unresolved words remain unclassified. The selected-meaning file passed
**22/22** selected cases. The final affected
selection, including additional soft/hard-reset and restaging assertions,
completed **172/172** selected cases green.
It covers packed and ordinary capture, distinct WORDs
sharing an OBJECT, unknown provenance, detached recall, native IDs, the normal
controller, structural checkpoints, compiled word loops, output ownership,
reconstruction, word storage, and batch isolation.

Two fresh full attempts stopped at the unchanged per-worker memory cap:
the first completed **2,521/4,667** cases with the default
256-case/16-file batches, and the second completed
**2,725/4,667** with 32-case/4-file batches. Neither recorded a pytest failure;
neither is a passing full run. The second run's active reasoning case
passed alone at **1.76 GiB**. Both affected worker
groups then completed **155/155** cases using
eight-case, one-file batches; the largest worker peaked at **5.78 GiB**.
This changes process lifetimes only, with no omitted tests, assertion changes,
or higher resource limits.

A subsequent eight-case full run stopped at
**389/4,667** because an existing category fixture temporarily wrote an XML
file inside `data/`. The source hashes matched again after fixture cleanup,
but the run correctly rejected the transient mutation. The new fixture
isolation probe was red. Eight temporary XML
creation sites across six test files now use the system temporary directory.
Their affected selection passed **44/44** cases,
including the existing guard that rejects actual source mutations. Model
assertions and the runner's source validation are unchanged.

The next full run completed **4,668/4,668**
cases with two failures and no resource or source-validation error. Isolated
workers exposed a peer-pipeline probe that inherited its eager gradient-anchor
setup from an earlier test, and a routing-credit probe whose linear-sum loss
is invariant under its addition operator. The peer fixture now initializes
the required anchors itself; the routing probe uses squared values to give
different pairings different costs, with a fixed seed and the same nonzero
gradient assertions. Both files passed **27/27** cases, each in its own fresh
worker.

The final frozen default selection passed with **4,668/4,668** cases completed
(1,039.56 seconds). It used
`DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python
test/test_report.py --batch-size 8 --max-files 1`, with the unchanged ten-worker
pool, 8 GiB worker cap, and 28 GiB aggregate cap. Peak worker footprint was
**7.19 GiB**, and aggregate peak was **11.33 GiB**.
The default 256-case/16-file grouping's memory instability remains a runner
residue in `todo.md`; smaller batches validate the same default selection.

This closes the anchoring review note only. General non-anchor and
nested-reference meaning, typed answer adapters, and controller unification
remain in `todo.md`.

### Query-phase evidence (September 18)

The rebased phase reviewer probes first failed
before the mask existed. The expanded
reviewer set then passed 23/23 in 166.9 seconds;
it includes real fullgraph forward/backward and
trace/eager-island negatives. The broader phase-affected selection passed
280/280, using three bounded fresh workers with a
1.56 GiB peak. This is affected-file evidence for the changed phase code, not
a claim of a new single-snapshot global run under the no-rerun policy.
See [Query phases](QueryPhases.md).

### Nested-retention evidence (September 18)

The rebased nested-retention reviewer probes first failed
(22/22 expected failures). The focused
selection passed 68/68 in 76.1 seconds at a 0.50 GiB peak.
The broader LTM, sidecar,
thought-history, query-occurrence and checkpoint selection passed 309/309,
using two fresh workers (73.5 and 28.9
seconds; 0.50 GiB peak). These are affected-file evidence after the historical
no-rerun composite, not a newly rerun global snapshot. See
[nested retention](NestedRetention.md).

### Shared query-work evidence (September 18)

The rebased reviewer run was red because the QueryWork module did not
exist. The focused reviewer selection then passed 13/13 in 20.1 seconds at a
0.40 GiB peak. The broader,
non-overlapping query/phase/taxonomy/expectation/history/retention selection
passed 372/372, with fresh workers
at 1.05 GiB, 0.50 GiB, and 0.40 GiB peaks. This is affected-file evidence
after the historical no-rerun composite, not a new single-snapshot global
run. See [shared query work](QueryWork.md).

The actual MPS sequence-training check passed: 877.81 seconds in `runEpoch`
for two batches, 1.24 seconds for construction and a 2.54 GiB peak aggregate
footprint. All 137 parameter tensors, including the predictor, were on MPS.
This used the current `eager` backend and was slower than an earlier 280-second
CPU test call under different run conditions. It is not a controlled speedup
comparison, a held-out utility result or a current supervised-answer throughput
measurement. GPU performance tuning remains open.

### Expectation-retention evidence (September 18)

The retained-estimate reviewer probe first failed as intended:
no observation occurrence could yet be
bound to the row-local expectation view. The implementation probe then passed
1/1. Its capacity and actual pending/packed writer
selection passed 10/10; this includes the rule
that the last free LTM slot belongs to the understood external observation,
not its forecast. The first broad pass correctly exposed an old two-observation
count assertion after a new estimate row made the durable history three rows;
the assertion now checks the explicit pair and observation-only predictor
view instead.

The final affected selection passed **236 passed, 5 slow-skipped**
(278.55 seconds, 1.06 GiB peak). It
covers structured prediction/lifecycle, all three observation writers,
consolidated LTM and checkpoint sidecars, origin compaction/nested retention,
typed prediction query consumers, and ordinary thought-history boundaries.
It validates retained occurrence ownership and detached checkpoint fidelity;
it does not establish metadata prediction, residual-policy credit, learned
utility, or throughput gates.

A subsequent full run reached 4,010/4,639 cases before the seeded depth-3
relative-end-state probe exposed a lifecycle regression.
The new writer had attempted to bind provisioning
rows while external observations were suspended; the provisioning wrapper
recovered the exception per text, leaving that parse lifecycle incomplete.
The reviewer probe was red; after gating retention
to actual external boundaries, the direct chain, slow global-LTM and depth-3
probes passed 4/4, and the affected LTM,
expectation, chain and thinking selection passed 192/192.
Generic LTM recurrence and attention now exclude
estimate rows; this is still ownership containment, not residual-policy or
learned-utility evidence.


*From `Testing.md` lines 933-3012, moved 2026-10-30.*

## Selected meaning and one controller (September 20)

**Review correction:** `288b56b` passed regression but did not complete the selected-meaning work.
Its standalone language codec violated the three-grammar architecture, and its
natural-wording experiment is not accepted learning evidence. The review
correction below removes that path. Grammar-owned wording remains open and
learned questioning utility remains unproven.

The original landing consolidates public and normal reasoning in `run_selected_thought`,
retires the other controllers and next-idea selector, carries typed
set/code/child answers, retains nested meanings through all three writers, and
adds a separate supervised codec, subsequently rejected and removed.
[SelectedMeaning](SelectedMeaning.md) specifies ownership, credit and limits;
[KernelRetirement](DevelopmentHistory.md#frame-kernel-test-migration) maps all 61 removed frame-kernel tests.

The final checkpoint correction passed: 7/7 selected
cases completed, exit 0. This includes native-width and legacy strict reloads,
the trained natural-language experiment and the real model's language
loss/optimizer/checkpoint integration. The broader affected run had already
passed the controller, row-local rewards, output, nested meaning, taxonomy and
retirement contracts. Neither run substitutes for the full landing gate.

The rejected language experiment trained 36 annotated examples at seed 19 for 240 Adam
steps, then checks held-out nouns, the parthood sense of “has,” “contains,”
converse roles, alternate-sense negatives, execution, generation and
recomposition. All relation wordings appeared in training; its programs were
synthetic, and its function vocabulary was restricted to `a`, `has`, `equals`.
These results do not demonstrate grammar-owned interpretation or generation.
The controller tests separately establish exact zero-init baseline
behavior, actual-work policy credit and a real optimizer update. This is not the
multi-seed learned reasoning utility study, which remains open; on FineWeb,
controller reward awaits residual credit.

The first full default run completed
4,630/4,630 cases in 1,029 seconds with three failures. One expected inline
candidate descriptions to be rejected before retaining their complete children;
two expected the retired repeated-presentation/parity loop. The updated tests
check no-write candidate formation, refusal to execute before occurrence binding,
retention without fact authority, one presentation, and no invented closure.
Their preserved controller contracts are linked in the retirement audit.
The implementation source was unchanged by these final test migrations.

The corrected boundary/presentation tests and normal-controller regression
selection passed: 42/42 cases completed, exit 0.

In the final **source-matched default run**,
**4,630/4,630 selected cases completed, exit 0**, in **1,026.3 seconds**.
Unique outcomes are 4,299 passed, 330 skipped and one expected failure; there
are no failures. The full selection used `--batch-size 8 --max-files 1`, ten
workers and unchanged limits (8 GiB per worker, 28 GiB aggregate, 1,800-second
worker and 10,800-second suite deadlines). Peak footprints were 7.19 GiB per
worker and 10.80 GiB aggregate. No selector or test exclusion was added.

This establishes the original regression result only. The review reopens the
work; expectation/residual credit, the generation-catalogue migration and the
full learned-utility study remain open.

## Three-grammar review correction (September 20)

The separate `LinguisticMeaningCodec`, its fallback and pre-generate route,
`WhatStepChooser`, `TruthInterval`, nearest-word realisers and the soft bridge
policy are removed. The sole controller now receives mode, polarity,
alpha-renamed bindings/scope and bounded attended visible STM/LTM content.
Incomplete-context checkpoint policies and their optimizer moments reset;
current policies restore strictly. The chooser can select the active question
as a nested `what(Q)` through its already-owned occurrence.

The depth-one/two probe fits the real MLP to authored routing examples, then
runs its selections and actual episode credit without a scripted chooser.
That is mechanism evidence only, not useful learned decomposition. The
natural-wording replacement must run through compose/generate on actual
forward text and a normal generation vocabulary; that gate is still open.
The matched-compute, multi-seed utility comparison remains open and gates
every learned-utility claim.

All 19 previously unexplained test deletions have individual dispositions in
[KernelRetirement](DevelopmentHistory.md#frame-kernel-test-migration), along with the tests removed or
migrated in this correction. The taxonomy public-entry probe now asserts the
bridge builder is absent and still forbids the global vector-read route.

In the final **source-matched full default run**,
**4,599/4,599 selected cases completed, exit 0**, in **1,029.4 seconds**.
Unique outcomes are **4,268 passed, 330 skipped and one expected failure**.
The selected and completed case sets are identical; no selector or exclusion
was added. The default selection includes the documented slow-test skips and
does not establish any open learning or throughput gate.

Command from `basicmodel/`:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

Ten workers used the unchanged 8 GiB worker / 28 GiB aggregate limits and
1,800-second worker / 10,800-second suite deadlines. Peak footprints were
4.46 GiB per worker and 12.56 GiB aggregate.


## Grammar-owned wording architecture (September 20)

`d1d8b5b` is an architecture/wiring increment, not completion of the work. Its
initial completion claim incorrectly moved the working natural-wording gate to
the learned-utility study. The user's clarification relaxed structural-preference and routing-share
measurements, not that original gate. At that commit language quality remained
unproven and the work remained open. See [SelectedMeaning](SelectedMeaning.md).
No additional LM, interpreter, realiser or auxiliary language loss is added.

The existing compose MLP now retains ordered operand and role context. A zero
input projection into its existing first hidden layer preserves old predictions
and initialization order; ordinary reconstruction/answer credit trains it.
BasicModel enables the declared generate walk and its existing supplied-answer
policy objective. The natural word `equals` is removed from the shipped
technical anchor tables. The ordinary forward program and output trace own the
actual choices; no hand-annotated `AnswerProgram` supplies the new text probe.

The text probe presents “a bicycle has a wheel” and “a wheel belongs to a
bicycle” through the real input/forward/reconstruction path and `runBatch`.
It checks live compose-MLP updates and generate-path invocation, while missing
answer labels leave generate-policy weights unchanged. It does **not** assert
that those sentences have learned canonical meanings or fluent outputs. The
existing supplied-answer output tests separately check actual policy updates.
All prior tests are retained; the new probes add coverage rather than replacing
language-quality evidence or retiring any further tests.


In the final **source-matched full default run**,
**4,608/4,608 selected cases completed, exit 0**, in **1,036.3 seconds**.
Unique outcomes are **4,277 passed, 330 skipped and one expected failure**.
Selected and completed case sets are identical. The full default selection
has no added selector or exclusion; its documented slow-test skips do not
establish the remaining language-quality, utility or throughput goals.

Command from `basicmodel/`:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

Ten workers used the unchanged 8 GiB worker / 28 GiB aggregate limits and
1,800-second worker / 10,800-second suite deadlines. Peak footprints were
7.19 GiB per worker and 10.80 GiB aggregate.


## Working grammar wording gate (September 20)

This completion keeps the original natural-wording gate with this work. A selected
`surface` attachment preserves the complete right semantic subtree while
retaining the marker numerically. Both attachment and wait decisions are made
by the existing grammar MLP; a small marker prior is computation inside the
operator. The normal generate grammar emits word concepts before the lexical
inverse compares against every known spelling. No LM or fallback interpreter
is introduced. [SelectedMeaning](SelectedMeaning.md) records the design and
the scope of the result.

The explicit slow run uses seed 931, 1,599 actual forward-parsed training
sentences, 1,000 generate-learning steps and 8,000 compose-learning steps. It
holds input encodings fixed for this focused grammar study. Annotated operand
variables get teacher-state full-code augmentation. There are 32 development
cases and 24 additional cases with three previously unused complete marker
wordings and eight previously unused nouns, in both orders. Result: 56/56
comprehension/control cases, 53/53 natural relation outputs and 53/53 meaning
recompositions. The three alternate `have` cases do not invent part relations.
All 178 observed word spellings compete in lexical inversion. This is a small
supervised compositional result, not general English, unsupervised learning,
or useful learned questioning. The causal utility study remains open.

The shipped XML validates against `model.xsd`. This ordinary CLI smoke exited
0 after one training batch (reconstruction loss 1.2483), and saved its normal
checkpoint. Its untrained diagnostic output is **not** language-quality evidence:

```sh
BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager BASIC_MAX_BATCHES=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python bin/Models.py data/MM_grammar_wording.xml
```

Reproduce the focused language gate with:

```sh
BASICMODEL_DEVICE=cpu RUN_SLOW=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py test/test_surface_grammar.py::test_real_text_has_a_complete_selected_meaning --batch-size 8 --max-files 1
```

In the final explicit slow run,
**1/1 passed**, 154.5 seconds, 1.24 GiB peak on CPU. It reruns all 56 cases, 53 generated
relations and 53 recompositions from fresh training at seed 931 on the final source.
The default suite skips this marked slow study. No existing tests are deleted.

In the final **source-matched full default run**,
**4,616/4,616 selected cases completed, exit 0**, in **1,039.3 seconds**.
Unique outcomes are **4,284 passed, 331 skipped and one expected failure**.
Selected and completed case sets are identical. No selectors or exclusions
were added to the default suite; the explicit slow run above supplies the
separate language-quality evidence.

Command from `basicmodel/`:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

Ten CPU workers used the unchanged 8 GiB worker / 28 GiB aggregate limits and
1,800-second worker / 10,800-second suite deadlines. Peak footprints were
4.46 GiB per worker and 8.70 GiB aggregate. Learned questioning utility
remains unproven.

The first full run completed all 4,615 cases
in 1,055.3 seconds with five failures. Two revealed that separate-stage
taxonomy configurations must retain their terminal registry owner; readers now
use the registry's actual owner. One exposed a structural taxonomy neighbor
without a numerical payload: the bounded reader now retains its unavailable
reference and reports incompleteness without allocating or aborting. One width
assertion still expected the symbol's content slice; it now checks the full
opaque concept width. The fifth required identical float32 top-k index order
across GEMM tilings. Seed 2385 reproduces a two-neighbor swap with distance
differences no greater than 2.4e-7. That test now checks unique selected rows and
distances against a float64 projective-distance oracle, with a 16-epsilon
absolute bound. No production ranking code changes. These are corrections,
not deleted tests or exclusions. The complete source is rerun after the fixes.

The focused correction run passed 41/41 cases,
including both taxonomy ownership/checkpoint failures, actual per-row controller
credit, the new missing-payload case, opaque grammar width and lexicon geometry.
The next full run was interrupted at 377/4,616 cases
(exit 130), with no observed failures, after a direct MPS check showed that the
new float64 oracle needed CPU storage. Its concurrent slow run was also stopped.
All workers exited before the test was edited. Only the reference calculation
moves to CPU; the lookup under test stays on its selected device.

The portable lexicon checks passed **27/27 on MPS**.

The next full run completed all 4,616 cases in
1,038.9 seconds: 4,283 passed, 331 skipped, one expected failure and one
failure. Its remaining failure exposed cancellation in the codebook's
straight-through estimator, `e + (q - e).detach()`: a small quantized value
can be rounded away even for unit-scale encoder values. A deterministic
probe reproduced it. The implementation now forms
the zero-valued gradient carrier first, `q.detach() + (e - e.detach())`,
preserving the quantized forward value exactly and the identity encoder
gradient. The retained test now checks those values, gradients and the absence
of target gradients on fixed cancellation cases. All 11 affected checks passed
on CPU and MPS. All full-run
workers had exited before either source or test was changed. Both the full
default suite and the fresh slow language study are rerun after this correction.


## Gradient factorization (September 20)

This landing replaces the global reconstruction-priority projection with ordinary
shared gradients and a measured boundary: output receives a detached concluded
idea and detached contextual inputs, while the generation operator's own
parameters train. Expectation still reaches current-step preceding encodings;
its target and previous-step context detach. The thought chooser's credit
trains its policy only. [Contract](GradientFlow.md).

At this landing, the production dictionary used the shared optimizer
(`conceptualContextLearningRate=0`); the September 21 corrections below restore
rotation ownership.
That landing's normal-batch diagnostic verified nonzero codebook reconstruction credit
and actual operator reconstruction/output overlap, in addition to standalone
cosine arithmetic. Sparse measurements never densify the inventory. The run
harness logs weighted R/O/E norms, R/O and R/E cosines and names persistent
opposition. No per-operator guard was justified by these short observations.

The first boundary probe failed both output paths on leaked conclusion-state
credit. The pre-removal diagnostic run passed 9/9 cases with the old
projection still present. Only then was the projection deleted. The recheck
covered corrected boundaries, the executed tied inverse and diagnostics.

The final explicit slow run passed both
real packed reconstruction/expectation studies with `RUN_SLOW=1` in 362.8 s.
The final **source-matched full default run** exited 0: all 4,609 selected
cases completed (4,277 passed, 331 skipped, one expected failure), 1,055.7 s
and 14.93 GiB peak aggregate memory. Command:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

`test_reconstruction_priority.py` becomes `test_joint_objectives.py`. Its
real packed prediction, exact reconstruction-family learning, shared ownership,
independent heads, sparse updates, AMP, one-backward/one-step, truth scaling and
closing-layout checks remain. The retired tests' dispositions are:

| Old check | Disposition |
|---|---|
| prediction_can_refine_a_reconstructable_representation; model_balances_prediction_and_thinking_without_answer_labels | Projection-specific synthetic checks replaced by the actual joint sum and retained real packed training checks. |
| fidelity_tolerance_uses_unscaled_loss_under_amp; invalid_reconstruction_tolerance_fails; invalid_ratio_fails | Deleted knobs reject configuration; ordinary AMP scaling remains tested. |
| projection_and_combined_norm_budget; zero_and_missing_reconstruction_keep_bounded_output_credit; enormous_opposing_output_does_not_erase_reconstruction | Projection/budget behavior is retired; ordinary sums and observational opposition reporting replace it. |
| large_finite_gradients_do_not_overflow_projection; sparse_projection_matches_dense_without_capacity_allocation | Large finite and sparse/missing/dense cosine checks moved to `test_gradient_factorization.py`; no gradient modification remains. |

The immediate-stop generation fixture proves the state cut and generator
credit. A separate real `lower` inverse fixture executes the shared operator
and proves its nonzero gradient; an unused operator is never counted as evidence.


## Accessible-mind effects (September 20)

This landing replaces descriptor scope strings with ten checked subsystems and
explicit effect targets. The ordinary controller commits detached effects;
structural reads stay live owned snapshots. Order-zero `part` leaves its
vector residual, while higher-order parthood emits bounded symbolic support.
The existing LTM writer owns per-role leaf-code tensor columns and their
checkpoint/compaction lifecycle. Cued `what` retrieves an old matching frame
into ordinary thought history; the chooser's recent-store scan is gone.
[Current contract and limitations](AccessibleMind.md).

| Coverage | Evidence |
|---|---|
| Grammar permissions, structural capability refusal, detached thought effects | `test_accessible_mind.py`, `test_thought_operation_catalog.py` |
| Meronymic residual and higher-order symbolic value, including missing paths | `test_accessible_mind.py`, `test_query_vp_boundaries.py`, `test_grammatical_query_vps.py` |
| Old-row cue retrieval, priming, scope, fan, familiarity and contiguity | `test_accessible_mind.py`, `test_query_contract_boundaries.py` |
| Exact leaf postings, forward-forest indexing, checkpoint, compaction and actual codebook remap | `test_accessible_mind.py`, `test_nested_retention_owners.py` |
| Normal `what` effect, detached knowing, retained frame ownership and no implicit recent-LTM read | `test_accessible_mind.py`, `test_thought_review.py` |
| Budget cutoff, nested return/finish, typed result retention | `test_normal_thought_controller.py`, `test_nested_retention.py`, `test_thought_checkpoint_credit.py` |
| Chunk trust and `.where` addressing, `infer` in both directions, D8's graded answer cost, the worked line's replay | `test_thought_infer.py` |
| Measured recovery by depth, training since storage and distinct-code chain length | `test_mind_generativity.py`; limits in [AccessibleMind](AccessibleMind.md#measured-limits), rates [below](#accessiblemindmd) |

No existing tests were deleted. Taxonomy fixtures now use higher-order symbols;
order-zero `part` has its own geometric checks. Structural fixtures carry the
geometry capability and expect live cloned compose inputs. Description
occurrences use a charged direct address, so one record of allowance reaches
an old row; zero allowance still rejects it. `lookup` checks a previously held
frame and first proves that an unseen row is inaccessible. The controller
trace includes its ordinary explicit conclusion when reduced read work leaves
room for that action.

The final distinct-code measurement passed all
22 selected measurement/index cases. At 0, 1 and 8 updates, a single code
recovered exactly; chains of lengths 2, 3 and 5 recovered at rate zero.
These are reported null results, not successful learned generativity.

The first complete default run completed all
4,632 cases with eleven failures. Old taxonomy fixtures used order-zero
points, two standalone fact fixtures supplied no index cues, and interface
checks still expected an empty write scope or an obligatory `what` child.
Those checks now cover the corresponding subsystem effects without deleting
cases. Public vector questions assert a meronymic result and no taxonomy
identity/proof; native taxonomy fixtures use higher-order symbols. Fact
support retains bounded occurrence/origin/text/trust provenance without
retaining the stored meanings as frames. The existing conflicting-source
check verifies both source texts and occurrence identities.

The first correction run completed 72 cases
and found one additional candidate-contract defect: a trained chooser could
select open `part` with an unnamed vector. Closed `part` accepts full-width
vectors, but grammar-open taxonomy enumeration now requires a native
reference consistently in formation, candidate construction and dispatch.
The new regression check asserts that no such invalid open candidate is offered.

The expanded correction recheck passed all
140 selected cases, including the public model entry, taxonomy provenance,
checkpoint restore, index and normal controller contracts.

The final **source-matched full default run** exited 0: all 4,633 selected
cases completed (4,296 passed, 336 skipped, one expected failure), 1,041.8 s
and 14.11 GiB peak aggregate memory.
Command, from the isolated BasicModel checkout:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

The full run repeats the distinct-code recovery measurement and reports the
same zero compound rates. Five optional `sentence.pt` embedding probes skip
in this isolated checkout because that locally built artifact is absent;
the main-checkout supplemental run below exercises the existing artifact.


The supplemental main-checkout run completed all 17 `test_testpoint.py`
cases with exit 0 (five passed, twelve skipped), including all five
embedding probes using the existing `sentence.pt`.


## Rotation-owned dictionary: review corrections (September 21)

The baseline includes documentation commits `7d4c290` and `e13c4ea`. The
dictionary correction was made and measured first, as requested. All 11
configurations changed by `7c2fa5a` restore
`conceptualContextLearningRate=0.01`: the shared concept dictionary is a
non-grad buffer, with the existing sentence-local unit-sphere rotation updater
as its sole owner. Objectives train operators and code-to-operation
projections; dictionary positions and `codebook.*` entries are absent from
the gradient diagnostic.

The prepared-answer boundary applies when `answerSynthesis=true`; the direct
head still trains the whole input state. Combining reconstruction and
supervised outputs without answer synthesis now warns at configuration, or
first training use if the data arrives later. The supplied-answer tied
benchmark now samples operator gradients every batch. Diagnostic exceptions
warn without aborting backward or the optimizer step; checkpointed opposition
streaks survive unused samples and reset on measured nonnegative agreement.

`QuerySignature` inherits checked `Subsystem` scopes from its canonical
thought descriptor. The allocator caches row-to-concept identities, restoring
the cache from its existing checkpoint owner. The leaf-code column grows
geometrically and stores its used prefix in the checkpoint tensor; reset,
compaction, remap and subsequent appends retain the existing index contract.
The current docs now describe ordinary gradients and
the state boundaries, rather than the deleted projection. `lxml` was added to
requirements only; no dependency was installed. The 11 XML files validate
under `xmllint`; pytest's optional `lxml` cases still skip in this environment.

### Fixed-seed reconstruction baseline

Probe, seed 42, CPU, one Torch thread,
Torch 2.14.0, native eager tensor loop, `data/MM_ladder.xml`, batch size 2.
Each run uses four validation batches, seven ordinary training updates, then
the same four validation batches. The training mean and rate exclude two
warmup batches. Both runs begin from the same seeded initialization and data;
the dictionary update setting is the intended configuration difference.
Checkpoint writes and compilation are disabled.

| Measurement | Optimizer-owned dictionary | Rotation-owned dictionary |
|---|---:|---:|
| Initial validation reconstruction | 0.100593256 | 0.100593256 |
| Training reconstruction, last five batches | 0.092587703 | 0.092793070 |
| Validation reconstruction after seven updates | 0.092143942 | 0.092326729 |
| Training sentences/s, after warmup | 0.56807 | 0.56798 |

The post-update validation shift is **+0.000182787 (+0.198%)**. This records
the changed baseline; it is not evidence of improved learning.

### Thought-enabled throughput

The fixed thought config declares
the complete grammar and enables the ordinary LTM owner. For each parsed
sentence, the probe executes one declared `quantize` request on a retained
leaf through the normal controller. This deliberately exercises thought
effects and retention; it is not learned question selection or a wording
gate. It measures native parsing, reading, appends, the thought call and the
epoch's post-tick compaction together. Two warmup batches precede ten measured
batches, comprising 20 sentences and 20 executed thought requests.

| Measurement | Before cache/capacity changes | After |
|---|---:|---:|
| Sentences/s with thought | 14.65773 | 14.64366 |
| Reconstruction | 0.102225167 | 0.102225167 |
| Executed thought requests | 20 | 20 |

The rate changes by −0.096%, a **null throughput result** on this small
workload. The amortized allocation and absence of a per-effect dictionary
scan are mechanism checks, not a demonstrated end-to-end speedup.

### Regression evidence and dispositions

All 112 affected cases passed, including the actual normal-batch operator
diagnostic, prepared-answer state boundaries and checkpoint/index lifecycle.

That native training batch measured `operator.CS.surface` reconstruction norm
0.000749390, output norm 1.829513384 and R/O cosine 0.087219789. Expectation
was unused in that batch and has a null cosine. This is an actual shared
operator measurement, not evidence about persistent agreement or useful
learning.

The mistaken selector run exited at collection
because `test_query_contracts.py` does not exist. The corrected affected run
above selects `test_query_contract_boundaries.py` and `test_query_registry.py`.

No existing tests were deleted. The real-batch optimizer-ownership assertion
is inverted: the dictionary must be a non-grad buffer outside all optimizer
groups, while shared operators still receive actual reconstruction/output
credit. Generic sparse cosine tests retain their numerical coverage under
operator names. Added checks exercise all 11 configs, direct-head warnings,
nonfatal diagnostics, integrated opposition checkpoint restore, checked
signature scopes, geometric growth and cached thought effects. The existing
structural checkpoint test also verifies the reconstructed reverse cache.

The first full run was explicitly interrupted at
3,228/4,651 completed cases with no observed test failure. Review found that
PyTorch serializes a tensor view's entire backing allocation: saving the used
leaf-column slice still retained spare capacity. All workers exited before
edits. The strengthened serialization probe failed
as intended (4,096 stored bytes for 3,072 logical bytes). Checkpoint export now
clones the used prefix, and the test exercises an actual save/load round trip
before appending. This checkpoint-only correction does not change the timed
workload, whose checkpoint writes are disabled. The corrected
checkpoint/index/retention run passed all 31 cases before the final full run.

The final default run exited 0: all
**4,651/4,651** selected cases completed (**4,319 passed, 331 skipped, one
expected failure**), 1,045.1 seconds and 13.06 GiB peak aggregate memory.
Command from `basicmodel/`:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

Optional slow learning studies are not part of this default run, and
the unavailable `lxml` schema cases remain skips. Nothing here closes the
remaining learning gates or starts the expectation work.

## Negative-image expectation (September 21)

This landing implements pure composition, the per-role signed image at the closing,
all-role surprise, observation/estimate retention, grammatical object masks,
and residual credit on the existing controller. The design, configuration and
checkpoint migrations are in [ExpectationRetention](ExpectationRetention.md).
The situation context and its three configuration variables are untouched.

The initial failing probes exposed absent negative-image helpers, erased
residuals on empty roles and missing durable surprise. Native exploration then
exposed two integration defects: `what` could bind a declarative description,
and a subsequent declarative input attempted to append a legacy parity slot to
ordinary thought history. The catalogue now uses known question occurrences
for `what`; ended observations stay with their existing observation owner.
The final generation smoke also caught raw text crossing the tensor input API.
Packed native exploration exposed the new object-mask lookup's one-record
limit: a later serial occurrence could not be read. It now scans the bounded
history on the shared meter and handles exhaustion through the controller's
normal cutoff. A separate delayed-contrastive probe found that replay had
covered MSE but not the optional contrastive objective; both now replay current
predictor parameters while retaining the original estimate as evidence.

Affected validation: **235 passed, 14 skipped** across the prediction,
retention, controller, output, migration, reset and retired-prior files. After
strengthening the absence probe and cleaning obsolete test descriptions,
**92 passed, 1 deselected** in the relevant focused selection. The native
unlabelled run includes three real `runBatch` calls, tied reconstruction,
backward and optimizer updates to the same chooser. No answer is supplied.
The generation correction passed **29 cases, 2 skipped, 1 deselected**.
After the final history and contrastive corrections, the focused prediction,
controller and work-boundary selection passed **71 cases, 1 skipped,
3 deselected**. The first full run was explicitly stopped before editing
source for these corrections.
The next full run completed all 4,665 cases and found two controller
regressions.
Reads could consume the last work unit before history recorded it; exhaustion
now records the already-paid work and enters the ordinary zero-cost drain.
The explicit routing-calibration fixture now initializes only its selected
feature subspace, so ignored columns cannot change its initialization. Its
800-update limit, actual learned descent, depth and credit assertions are
unchanged. Cutoff coverage now runs both with and without a prior comparison.
The corrected controller selection
passed **80 cases, 4 deselected** before the final full run.

Learning is measured separately
from these mechanisms. Three-seed frozen synthetic and parsed-English studies
beat shuffled/context-free prediction controls. The existing wording recipe
again parses all 56 held-out examples and generates/recomposes the expected
meanings. Related unexpected text leaves a smaller conceived remainder in
120/120 comparisons. Residual policy updates run without labels, with their own
baseline and measured return variance.

Six native runs cover unpacked and packed input, each with prediction plus
queries, prediction alone, and reconstruction alone. The packed runs each
complete 78 observations and 74 pairs in seven optimizer steps. None beats its
context-free prediction control; none demonstrates joint quality benefit.
Native reconstruction uses a 16-row basis and still reports truncated rows,
so it does not establish full-basis fidelity or discrimination.

**Reasoning utility is null:** at gains 0 and 1, held-out questions take the
same two thought steps and 17 work units, and have identical unsuccessful
positive-assertion answers (Brier score 1.0). This does not pass the useful-query
or reasoning-work learning gate. The remaining joint/causal learning evidence
stays with the expectation work in `todo.md`; it is not transferred to the
learned-utility study.

Retired assertions and their replacements are dispositioned in
[ExpectationRetention](ExpectationRetention.md#migration-and-regression-dispositions).

Final source-matched default selection:
**4,338 passed, 330 skipped, 1 expected failure, 1 compiler-cache failure**;
all **4,670 selected cases completed** in 371.338 s. The sole failure was
`test_normal_batch_trains_supplied_grammar_lessons`: clang rejected a generated
precompiled header whose recorded PyTorch header timestamp predates the
installed header. It was not an assertion failure. The old generated-header
directory was preserved as a backup so compilation can rebuild it. No package
or repository source was changed to address the environment error.
The affected case then passed in 185.99 s
against the same source after rebuilding the generated headers.

The user explicitly waived another full rerun: **“No need for the full rerun.”**
This is a completed full run with a documented exception, not a claim of a
green full run.
The runner used its default 256-case/16-file batches, 10 one-thread workers,
8 GiB per worker and 28 GiB aggregate cap; peak aggregate memory was 18.83 GiB.
Optional slow learning studies are separate; unavailable `lxml` checks
remain skips.


## Retention score and pair lookup: review corrections (September 21)

This correction repairs the expectation landing's retention score and pair
lookup. The retained scalar normalizes over the soft union of observed and
expected roles, so equally surprising idea and relation rows score equally;
the predictor still trains against the full all-role residual. Native purity
checks now include nonuniform priority and live reading scope. Indexed pair
reads are checked after restore and compaction. The harness performs one
visible fresh-worker retry for a typed stale precompiled-header failure, with
PCH reuse disabled and compilation still active; genuine test failures remain
failures. Operator reports now include both norm ratios beside cosine.

The affected run completes **131/131** cases: **128 passed, 3 skipped**,
exit 0. The full run completes
**4,683/4,683** cases: **4,352 passed, 330 skipped, 1 expected failure**,
exit **0**, in **1,059.92 seconds**, with **no waived failure and no cache retry**.
The previously failing grammar-lesson compiler case passes. These are unique
case counts.

The full run used `--batch-size 8 --max-files 1`, ten workers, the unchanged 8 GiB
per-worker / 28 GiB aggregate caps, and a 14.34 GiB peak aggregate footprint.

The native gradient report
measures output/reconstruction norm ratio **2,441.3373** for
`operator.CS.surface`, with cosine **0.08721979**. This is one diagnostic batch,
not a learning gate. The earlier null useful-query result remains open.

## Generation ownership (September 21)

The September 17 generation candidate is rebased onto the shared-operator
contract. Both output modes retain a declared generation catalogue, ordinary
output inverse dispatch uses its model's scoped catalogue, and numerical hosts
keep one optimizer owner and their existing checkpoint names. Chooser weights
and Adam moments retain rule-meaning migration. The concluded idea remains a
detached operand for output; supplied output trains the active conditioner and
any shared maps actually executed. No independent operator copy, objective or
global gradient budget is installed.

The non-loop fixture's shared-map credit is absent. This
landing validates mechanisms and normal supervised updates; useful learned
language, expectation and reasoning remain open.

The broad affected run passes **212 cases, 18 skipped**; explicit slow
supervised integration passes **4/4** on MPS. After correcting the compatibility
and adapter fixtures, the final affected run completes
**76/76** cases: **60 passed, 16 skipped**, exit 0.

The full run completes **4,695/4,695** cases:
**4,364 passed, 330 skipped, 1 existing expected failure**, exit **0**, in
**1,059.34 seconds**, with **no waived failure and no cache retry**.
The run uses `--batch-size 8 --max-files 1`, ten workers, the unchanged 8 GiB
per-worker / 28 GiB aggregate caps and a **14.73 GiB** peak aggregate footprint.

An existing untrained MentalModel overflow reproduced at seed 3 with identical
inputs and scores on that landing's baseline and runtime. This landing pinned the
compatibility fixture to a successful seed and separately asserted rejection
of the failing initialization. **The seed audit removes that workaround:** consumed
operands were reused by recursive soft compaction. See the correction and
seed-selection audit below.

Final documentation-link run passes **69/69** cases, exit 0.

## Reconstruction baseline and seed audit (September 21)

A seed may make a measurement reproducible; it cannot be selected to make a capability
assertion pass. Known failing initializations remain useful regression cases
beside unseeded compatibility assertions. Convergence failures remain failures,
with no replacement passing seed, relaxed threshold or new expected-failure
marker.

The corrected compaction matches exhaustive small tilings and enumerated
derivatives, including a fullgraph capture. The known failing MentalModel
seed 3 now forwards successfully beside an unseeded smoke test. The declared
32-seed measurement passes 32/32. The native fixed-seed reconstruction baseline
is exactly unchanged; the tied packed/single comparison is a null for parity,
with mean costs **.7866926491 / .6838697642**.

The bounded Pi/Sigma pair's hidden tanh is cancelled by the next atanh;
its old seed-selected XOR assertion is replaced by an interior algebra check.
The adjacent exponential-feature XOR learning gate retains its original bar.
The explicit slow audit exposes a grammar-XOR miss, zero free-derivation
recovery and two unsupported W=6 configurations. Those failures remain
open. A default run does not close these learning gates.

The final source-matched full selection completes
**4,706/4,706** cases: **4,375 passed, 330 skipped, 1 existing expected failure**,
exit **0**, in **1,064.73 seconds**, with **no waived failure and no cache retry**.
The run uses `--batch-size 8 --max-files 1`, ten workers, unchanged 8/28 GiB
caps and a **10.78 GiB** peak aggregate footprint. The corrected affected
selection passes **48/48**.

Final documentation links pass **70/70**.

## Concept parts and provisional rows (September 22)

The implementation
covers the live union default, optional
conjunction/union passes, signed exponents, provisional row pool, context
weights, detached use gate, transpose reverse and checkpoint/optimizer
ownership. The host promotion dictionaries and additive conceptual hop are
removed. Object testimony and sequence removal remain open.

The layer retains presence .9999998808 through four orders with one of eight
parts present. Thirty-two weak parts at .001 give .0315085351, below the
stated .031 excess bound. Balanced transpose checks recover conjunction parts
at .9 from .81 and union alternatives at .6 from .84, within 1e-6. Context
matches assign a provisional pair immediately; use promotes it in place,
non-use permits recycling, and the optional co-present whole requires all
its parts. Both matrices and their state survive a fresh model checkpoint;
sparse growth/pruning preserves optimizer moments and pending credit by edge
allocation identity.

The affected selection
completes **140/140** cases, **133 passed and 7 skipped**, exit 0. The explicit
unseeded XOR selection
completes **3/3**, exit 1: the four-conjunction signed representation passes,
but four and eight zero-initialized conjunction concepts both finish 900 Adam
updates at **MSE .5**, against the unchanged **< .1** bar. There is no seed
selection, retry of an initialization, relaxed threshold or expected-failure
marker. This learning residue stays open.

The repeated `d4dc385` native reconstruction means are unchanged with the
switch off and on. The native workload is serial and does not exercise the
sparse pyramid; trained sparse-pyramid reconstruction remains unmeasured.
The packed/single payloads remain unchanged, including the .7866926491 /
.6838697642 mean tied byte costs. Diagnosing that gap is left to the
packed/single parity landing (September 25).

Scratch XML fixtures now live outside the validated source tree, with a
regression checking the snapshot while a scratch file exists. The runner's
source guard, finite limits and coverage requirements are unchanged.

The source-matched full selection completes
**4,717/4,717** cases: **4,384 passed, 332 skipped and one existing expected
failure**, exit **0**, in **1072.14 seconds**. There is no waived
failure and no compiler-cache retry. Peak aggregate footprint is **10.63 GiB** under
the 20 GiB cap, with 8 GiB per worker. The two new default slow skips are
explicitly measured XOR failures, not capability passes.

Final documentation links pass **71/71**.

## Paired evidence: review corrections (September 23)

The implementation
replaces the half-present snap background with independent `(c⁺, c⁻)`
evidence per occurrence. Sigma and pi compute dual folds, with nonnegative
exponents and explicit source polarity. Both survives independently of
neither through the transpose, thought effects, symbol leg and conceptual
checkpoint. Context matching uses a dense host tensor over the taper span;
alternative exponents normalize by their maximum, discovered gates stay at
one, and sparse values have one checkpoint owner.

The production-snap regression reproduces the old `.75 / .9375 / .99609375`
false presences with 2 / 4 / 8 inactive parts and now reads zero throughout.
The trained parallel smoke model supplies a measured `.005` admission floor;
independent feature-permutation controls are rejected at 8, 64 and 256 slots.
The live eight-slot batch retains weak evidence below the discovery use floor.
This is a bounded calibration and live measurement, not learned XOR or
successful discovery. The `d4dc385` serial reconstruction means match exactly
with pi off and on. Packed/single parity remains open.

The final affected selection
completes **247/247** cases: **219 passed, 28 skipped**, exit 0. The explicit
slow selection completes
**37/37**: **35 passed and two strict expected failures**, exit 0. Both
unseeded zero-definition XOR gates retain MSE **.5** after 900 updates and
the unchanged **< .1** learning bar. The expected-failure disposition is
the September 23 review decision; the primitive-properties landing below owns
the end-to-end learning gate.
The paired LTM closing and object testimony remain open.

The final source-matched full run
completes **4,743/4,743** cases: **4,410 passed, 332 skipped and one existing
expected failure**, exit **0**, in **1,061.76 seconds**, with no waived
failure or compiler-cache retry. Peak aggregate footprint is **17.003 GiB**
under the unchanged 20 GiB aggregate / 8 GiB per-worker caps.

The final documentation-link check
completes **72/72** cases, all passed, exit 0.

### Primitive properties and grounded extents (2026-09-23)

The implementation
replaces the canonical WholeSpace tag lookup with learned memberships over
byte primitives. Its eight supplied properties reproduce all 256 byte
memberships exactly, including the four requested properties; all 160
constant-signature runs in the 32-sentence ladder idiom fixture are unchanged.
Independent tower codes meet through one input extent and retain their
positions through concept composition, symbol readout and checkpoints.

The new native XOR learning assertions use six unseeded runs, pool sizes four
and eight, and four actual conjunction definitions in each. Primitive naming
receives 900 updates and paired XOR supervision 600. The native tower folds
supply OR and AND; negative parts start unwritten and are learned. The gates
pass in both the affected selection and fresh full-suite processes, whose
paired errors range from
.00001935 to .00002694. This is supervised learning, not an unsupervised
concept-discovery result. The old strict-expected-failure pyramid assertions
are replaced.

The affected selection
completes **187/187**: **178 passed, 9 skipped**. The explicitly enabled
slow selection completes
**12/12**, all passed. The final full run
completes **4,767/4,767**: **4,436 passed, 330 skipped, one existing expected
failure**, exit 0, in **1,083.18 seconds**. No failure is waived and no
compiler-cache retry ran. Ten one-thread CPU workers use eight-case,
one-file batches, with 8/20 GiB worker/aggregate caps and a 16.99 GiB peak.

The corrected serial reference reader restores the exact `d4dc385`
pre-training reconstruction mean. Learning the new memberships changes the
training-window mean by +2.28% and the after-training mean by +.142%
(.0924575571 versus .0923267286); freezing just those memberships reproduces
all old means exactly. Pi off/on are identical in this serial control.
The calibrated .112 floor uses unit-ball units and permits native evidence
above the discovery threshold. Its synthetic controls remain below that
threshold at eight slots but accumulate false support at 256; this is a
bounded calibration, not a universal absence detector. The normalization
lifetime remains a review decision.

The final documentation-link check
passes **74/74** cases, exit 0.

### Located parts and pervading wholes (2026-09-24)

This correction closes
the residue of the first membership-read landing.
Word definitions reference one native fused PartSpace group, with its
ordered constituents retained in that tower. A part may occupy a proper
subspan; a WholeSpace property must pervade the occurrence. Different
witnesses supply alternative percept conjunctions on ordinary concept rows,
and survive meronomic refinement. The corrected slow smoke guard requires
positive evidence at position, extent and symbol scope after one epoch.

All **186 affected cases pass**, including explicit slow smoke and optimizer
checks and six unseeded native XOR runs at pool sizes four and eight. The
serial comparison is unchanged from the first membership-read landing: after-training reconstruction is
0.142% above `d4dc385`, and freezing primitive memberships reproduces that
reference exactly. Prior memberships and all 160 runs over 32 sentences
remain unchanged. Normalization is limited to assigned provisional
disjunctive rows; the accepted primitive-properties residue is closed.

The full run
completes **4,782 unique cases: 4,451 passed, 330 skipped, one existing
expected failure**, exit 0.


### Recurrent parts and attended rows (2026-09-24)

The correction covers recurrent part
formation, containment over canonical ids, and concept fields rebound per
sentence. Twelve distinct words are read through eight order-0 slots, both
across turns and in one batch. Checkpoints, symbols, priming and thought
retain concept identity across those bindings. The landed negative-pole
formula, witnessing, location and XOR gate are preserved; located XOR and
native perception are not part of this landing.

All **44 affected cases pass**, including the explicit slow optimizer and
training smoke checks. Serial reconstruction remains unchanged from the membership-read landing:
the after-training mean is 0.142% above `d4dc385`; frozen primitive
memberships reproduce the reference exactly. All eight prior rows over
256 bytes and the 160 runs over 32 sentences are unchanged.

The final full run
completes **4,794 unique cases: 4,463 passed, 330 skipped, one existing
expected failure**, exit 0. All six fresh unseeded XOR runs pass.


### Extent containment and restored raw smoke (2026-09-24)

The extent correction
fixes PartSpace containment at the subject
extent and raw analysis regions extending into padding. A subject shares
its contained parts across its positions, while WholeSpace retains pervasion
at each position. Separate extents retain separate evidence. Uniform raw
carrier regions now stop at the last observed byte; padded regions are empty.

All **129 affected cases pass**, including explicit slow tests. The smoke
guard again runs `MM_sparse_concept.xml` through a real optimizer epoch and
asserts positive evidence and exactly zero negative evidence for every
present word at extent, retained position, carrier and symbol scope. Each
of the four inputs reads `(1,0)` at extent and symbol scope. The six unseeded
native XOR runs pass; the accepted recurrence, attended-field and identity
behavior is preserved.

Serial reconstruction is unchanged from the accepted membership-read landing: the after-training
mean remains 0.142% above `d4dc385`, and frozen primitive memberships
reproduce that reference exactly. The eight prior rows over 256 bytes and
segmentation of 32 sentences into 160 runs are unchanged.

The full run
completes **4,798 unique cases: 4,467 passed, 330 skipped, one existing
expected failure**, exit zero. All six fresh unseeded XOR runs pass again.
The bounded run takes 1,092.1 seconds with a 12.39 GiB aggregate memory peak;
no compiler-cache retry ran.


### Located XOR and native perception (2026-09-24)

The current XOR gate learns a row's positive pole from primitive input.
Four unlabelled located cases are formed by pi at order 0 and symbolized;
a zero-initialized sigma row learns which cases are XOR at order 1.
Pools four and eight each have three unseeded runs, with no run selection.
The gate asserts nonzero initial error, the final positive pole, and that
no pi edge exists above order 0 or in the symbolic loop. A pooled both
corner cannot satisfy this gate. The classical DNF expression is a
composition check over retained positions only.

Unrelated content must remain exactly zero at position, extent, carrier
and symbol scope. Other affected probes cover per-pole witnessing,
unwritten inventory addresses, getter purity, transient field identity,
nonzero pass-back focus, grammar reference resolution and checkpoints.
The native priors and constant-signature segmentation are checked against
the primitive-properties landing. Serial reconstruction is measured against `d4dc385` and reported
without a tolerance gate.

The fixed serial measurement gives after-training reconstruction
**0.0891618710**, against **0.0923267286** at `d4dc385` (3.43% lower in
this comparison). Pi off, pi on and frozen priors agree. All eight prior
rows over 256 bytes and 32 sentences / 160 segmentation runs are unchanged.
This measurement sets no tolerance and does not close the separate
learning gates.

The final source-matched full run
completes **4,812 cases: 4,485 passed, 326 existing skips, and 1 existing expected failure**, exit zero. The final correction selection completes
**98 cases: 80 passed and 18 existing skips**, and the explicit relevant slow selection passes
**19/19**, including the restored raw smoke.
All six unseeded XOR runs improve from MSE .5 to 0.


### Refinement residue and normalized-means evaluation record (2026-09-25)

The refinement implementation
uses actual retained witness spans, preserves extent containment, removes
count-triggered raising, and gates promotion beyond order 1 on discontiguous
both support after local refinement stalls. Three completed updates without
strict local improvement was accepted by Alec on September 25.

All six native XOR row-learning runs still improve from MSE .5 to 0. The
normalized raw-Sigma/current-Pi pair solves all 16 declared unit-energy
measurement trials at widths 4 and 8; all six additional explicit learning
assertions pass. The recommendation was **drop as a production replacement**;
Alec accepted it and the evaluation mode and its switches are deleted.
Native min/max perception and the
max concept pyramid remain unchanged.

Serial reconstruction is .0891618710 after training versus .0923267286 at
`d4dc385`, without a tolerance gate. Pi off/on and frozen priors agree.
All eight prior rows over 256 bytes and all 160 segmentation runs across
32 sentences are unchanged. The native smoke has positive evidence and no
negative evidence for every present word.

The full run
completes **4,829 unique cases: 4,502 passed,
326 existing skips and 1 existing expected failure**,
exit zero. The affected selection passes 140 cases and the final fixture
correction selection passes 34/34. The explicit slow selection
is **27 passed and one failed**. Its inherited `XOR_exact` CLI crisp-output
gate fails at .61838 unseeded; untouched `606683a` also fails. At measurement
seed 42, baseline, current defaults and selective normalization report the
same predictions and MSE .27634. Its unary tower-fold hosts were deleted with
native perception, so the normalization switch cannot affect that fixture. The assertion
remained in place. Alec required the gate to pass through native conceptual
space; the review resolution below replaces this failed configuration.
No skip, expected-failure marker, or selected initialization was added.

Three later fixture corrections are
direct GrammarLayer inheritance with an auxiliary mixin, and distinct known
codebook rows for the relation-insertion probe, and omission of deleted
WS count-raise state from checkpoints. The final documentation-link check
passes **82/82**.


### Normalized means dropped and native CLI XOR — historical (2026-09-25)

This resolution
precedes the final review. Its saved native perceptual views
were rejected; the forward-artifact correction below supersedes that inverse.
It implements Alec's initial three decisions. Refine-before-raise keeps three completed
optimizer updates without strict improvement of the worst local both residual,
reset by improvement and cleared by pure/unknown evidence. Normalized means are **decided:
dropped**. The normalization mixin, lift/lower arguments, XML/schema parameters
and mode tests are deleted. The reasons: composed operands exceed unit energy, depth
loses spread, the geometric floor has a dead gradient below and steep slope
above, convex bias breaks all-zero input, reconstruction is worse and there
is no consistent speedup.

`XOR_exact.xml` now uses symbolic order 1, conceptual pi and native aligned
perception. Two primitive positions lie in one extent. Primitive/name lessons
teach “is a one”; the ordinary unlabelled observer forms located pi cases at
order 0. A zero-initialized order-1 sigma learns the XOR cases, and the output
reads that concept's positive pole by persistent identity. No pooled both
corner or pi edge in the symbolic loop supplies the result. Grammar owns
`lift` and `lower`; no unary tower fold is restored. The understanding retains
native perceptual views for the ordered inverse. A serial derivation or free
inverse cannot substitute the latest live field for owned reconstruction.

Both existing end-to-end CLI tests pass unseeded with unchanged function
bodies, decorators and thresholds. All nine declared native learning runs return
`[0, 1, 1, 0]` from initial MSE .5. The three added native curriculum runs
reconstruct all four inputs and assert exact-zero unrelated controls.
The restored native smoke and serial ownership checks pass with slow tests
enabled.

The single final full run
completes **4,818 unique cases: 4,497 passed, 320 existing skips and one existing
expected failure**, exit zero, with no red outcomes. The affected selection
completes **423 cases: 361 passed, 61 existing skips and one existing expected
failure**. The two CLI gates pass **2/2**, and the explicit slow selection
passes **23/23**. A pre-correction full run was interrupted; it contributes
nothing to these final counts. A memory-limited slow attempt was rerun with
one worker. No new skip or expected-failure marker is added.

Serial reconstruction after training remains **.0891618710**, against
**.0923267286** at `d4dc385`, with no tolerance gate. Pi off/on and frozen
priors agree; all eight prior rows over 256 bytes and all 160 segmentation
runs over 32 sentences are unchanged. Changes remain uncommitted and unpushed
for review; no parent bump has been performed. The final
documentation-link check passes **83/83**.

### Forward-artifact reconstruction: final review correction (2026-09-25)

The final correction
replaces the saved-view inverse. The native understanding retains concept ids,
paired field evidence, occurrence evidence and brackets. `_reverse_body`
resolves those identities and uses `cs_percept_attribution` to reach native
memberships and radix rows. Radix activity decodes the input; no saved event,
input stack, code-neighbour match or later live perception supplies the read.
An equal-length subject extent locates an ordered part group even when the
WholeSpace runs are finer. Property attribution follows written memberships;
an absent primitive receives no inverse gradient.

The initial CLI attribution attempt reconstructed **1/4** and failed crisp
output. The extent-placement and absent-membership-gradient probes exposed
the defects above. The corrected CLI reconstructs **4/4** and passes MSE < .05.
The reconstruction test's name said fifty percent but its old assertion
checked twenty-five; it now enforces the stated **50%** requirement directly.
No seed is selected and no learning-rate, epoch or tolerance adjustment is
used. Poisoned-access tests preserve a prior inverse after another perception,
and zero retained evidence reconstructs no bytes.

The mode-exclusion tests assert that serial grammar executes no field sigma,
pi or not, and parallel perception dispatches no grammar lift/lower. The
grammar category-learning smoke fixture now runs in serial mode, preserving
its assignment and role-learning assertions. `XOR_exact.xml` removes its
unexercised grammar and says that the lesson teaches only memberships and one
name, never XOR; reconstruction is best effort from forward artifacts.

The serial measurement remains **.0891618710** after training against
**.0923267286** at `d4dc385`, without a tolerance gate. Pi off/on and frozen
priors agree. All eight property rows over 256 bytes and all 160 runs
over 32 sentences match their preserved reference.

The final source-matched full run completes **4,824 cases**: **4,502 passed**, **321 skipped** and **1 existing expected failure**, with no red outcomes. The new slow serial exclusion check is included in the explicit passing selection. Alec approved publication on September 25; implementation `99207a3` matches all 648 reviewed source blobs.

Final documentation-link verification: **84/84 passed**.


## Packed/single reconstruction parity (September 25)

This landing diagnoses the
current-tree gap before the expectation-learning experiment. Actual endings and
leaves are identical on the declared four-sentence workload. Packed inverse
search and byte scoring previously admitted neighboring sentences' candidates;
saved programs also read an unfilled final root slot. Candidate ownership now
follows sentence ids, and the final closing completes the saved-state bank.

Earlier packed-answer measurements, including the September 16 supplied-answer
runs at batch size 2, reconstructed from zero-filled saved states instead of
the sentences' actual final states, so their answer-quality numbers are unreliable
and must not be cited.

Parity uses atol 1e-6 and rtol 1e-5 for every sentence, with three ambient
initializations in the regression and the historical seed 42 only for the
measurement. Every measured closing, saved program root, retained/recovered leaf
and byte cost agrees exactly. Mean byte cost is **.6839025617** in both layouts;
the native serial baseline remains **.0891618710** after seven updates.

The initial parity review's source-matched full default run completes **4,829 cases**:
**4,504 passed, 324 skipped and 1 existing expected failure**. The default
affected selection is **62 passed / 39 skipped**. The slow-enabled affected
audit covers all 101 cases across bounded runs: **100 pass and 1 fails**. The
nonzero compose-gradient assertion in the reconstruction-cache probe fails
on both the patched and committed runtime, before its cache assertion; that
failure remains open. Both inherited trace-fixture failures were
reproduced before correcting their operation preconditions without selecting
a seed or weakening their assertions.
The operand-provenance fixture now retains leaves until binary consumption;
that particular probe no longer covers a unary rewrite followed by a binary fold.

Expectation-learning utility remains unproven and belongs
to a subsequent landing; the accepted review correction below owns publication.

Initial parity documentation-link verification: **85/85 passed**.

The bank-contract review follow-up
investigates first-sight inventory growth and removes silent reconstruction
fallbacks for missing staging or a sentence with no usable candidates. Its
full run completes **4,839 cases: 4,514 passed, 324 skipped and 1 existing
expected failure**, with no red outcomes. The affected selection is **125 passed /
68 skipped**; the explicit slow selection passes **24/24**, including compiled
reconstruction and all three ambient parity cases. Packed/single measurements
and the serial baseline remain exactly unchanged.
Documentation-link verification passes **87/87** for that review snapshot.

The accepted host-sync correction
keeps the bank invariants on device with asynchronous assertions. Fake-tensor
and native MPS profiler probes reproduce the previous scalar host reads and
require their absence after the fix. Alec accepted Claude's review and
authorized publication of the full tree after this correction and validation.
The final source-matched run completes **4,844 cases: 4,519 passed,
324 skipped and 1 existing expected failure**, with no red outcomes. The
affected selection is **129 passed / 68 skipped**; all **24 explicit slow
checks pass**. Packed/single means remain **.6839025617** and the serial
after-training baseline remains **.0891618710**.
Final documentation-link verification: **88/88 passed**.

Implementation `0e70001` matches all 650 reviewed source blobs; the shared-fields landing below follows this parity landing, while the expectation-learning gates remain open.

### Shared fields, interpretation and fixed capacities

Serial and parallel passes share definitions and
checkpoint identities. Native percept poles pool inside one field bracket
before concept folds; events retain inverse attribution. The `interpret`
operator runs before every serial word composes, owns provisional testimony
and its reverse, and supplies grammar-selected references to the closing.
The original schedule replayed N serial sentences in parallel and then applied
label read-back. That order is superseded by the parallel-first follow-up;
ordinary serial reading now supplies the symbolic processing. All perceptual codebooks have fixed physical
capacity with logical admission and stable where-space slices.

The three-seed erosion measurement completes all nine conditions. It supports
the lower native FineWeb CP and higher within-category distance under parallel
training, but does not establish the full predicted erosion/restoration
mechanism. Native XOR CP is zero throughout this short measurement, and
FineWeb label read-back lowers CP in all conditions. These findings are
not learning-test failures to tune away. The native XOR
learning checks remain separate, with their truth and error assertions intact.

Fixture migrations follow the changed contracts: locations belong to native
events, physical reserves must fit the initial alphabet, and testimony reads
its naming word instead of ATOM/UNIVERSE bounds. META associations are
structural; the history-projection test targets the object row that references
the word. The former refinement fixture already had an exact answer under
native negative evidence, so it now learns a missing predicate link while
retaining its initial-loss and refine/raise requirements. Seeds and tolerances
are unchanged.

The original reviewed full sweep completes **4,888 cases: 4,562 passed, 325 skipped and one
existing expected failure**. The final affected selection passes **46/46**,
the explicit slow selection **24/24**, and erosion/native-boundary checks
**3/3**; all 34 changed XML configurations pass schema validation. The final
review run also passes all **91** documentation-link checks. Claude accepted
this original implementation for commit.

### Parallel context first

`interleave:N` reads the coming N complete sentences in native parallel mode
before their serial reading. There is no separate label read-back. The old
interleave rows are void for this order; the retired three-seed experiment and
its findings remain in FutureWork. The fixed-probe distance reducer stays
as a logged measurement with no pass/fail ordering.

The serial seven-update comparison is re-baselined at the reviewed shared-fields source:
mean training cost **.0949082300**, after-training cost **.0882481802**. The
previous values were .0879226878 and .0891618710. The fractional-gradient fix
changed the training trajectory: insertion credit now applies only to an
unwritten edge, while an existing fractional edge uses its true exponent
derivative. This is a new comparison baseline, not a claim of improvement.
The follow-up rerun reproduces all three phases exactly. Packed and single
sentence reconstruction also match exactly at **.6839025616645813**.

The final follow-up run
completes **4,905 unique cases: 4,580 passed, 324 skipped and one existing
expected failure**. The affected selection passes **110/110**, and the explicit
reconstruction selection passes **4/4**, including three ambient initializations.
No skip, expected-failure marker, selected seed or tolerance change was added.

Two earlier full attempts stopped at the worker memory cap. The reasoning
fixture now collects completed cases' model callback cycles; its entire file
passes in one worker at 3.13 GiB, and the final full run passes under the same
8 GiB per-worker limit. The work remains uncommitted, awaiting Claude's review.

Final documentation-link verification: **93/93 passed**.

### Shared-field corrections and the expectation-learning comparison (September 26)

This correction pass
covers all four corrections. The parallel context pass now runs without
gradients or an optimizer step before serial training. `interpret` reuses the
object already associated with a word, including a kind; order 1 is the
default only when a new object is needed. The larger physical
reserves needed for the intended vocabulary and corpus are recorded as open work.

The fourth correction restores the spatial and temporal bands themselves.
One model-owned `.where` ladder spans all preallocated input, part, whole
and symbol ranges, and captured programs carry that band. Its periods derive
from total capacity; an integer address is decoded when needed. One shared
`.when` ladder spans LTM capacity, and all percepts in a field share one time.
The exact clock remains available. Concepts retain their full content width;
the later closing work still owns writing field coordinates into LTM rows.

The source-matched serial seven-update baseline is **.1505906619 before,
.1347484022 during and .1362805218 after training**. The corrected coordinate
bands change the reconstruction objective's event inputs and the resulting
training trajectory. The later occurrence/time correction below isolates the
extra initial .05 as the cost of reconstructing the shared timestamp on every
word. These values are historical; use the new baseline below for subsequent
comparisons.
The serial baseline never enters the parallel context pass. Packed and single
sentence reconstruction still agree exactly at **.6839025616645813**.
Earlier packed-answer quality numbers, including September 16 batch-2 runs,
remain invalid because the saved programs held zero roots rather than the
sentences' completed states.

The declared learning comparison finishes
all **12 native arms: three seeds, four conditions, 64 optimizer updates**
per arm, with 16 identical held-out batches before and after. Ordered
prediction does not beat both trained controls on feature MSE and presence
BCE across seeds. Its FineWeb discrimination is worse than reconstruction-only
in every seed; byte reconstruction also loses that comparison in seed 1.
No run reaches the 512-candidate basis limit, but incomplete reverse programs
remain in the losses, with unavailable operators recorded. Each native
thought comparison has 25 unsupported question pairs: the meanings have no
grammatical VP, so answer error and work are unmeasured.

The separate frozen-text study retains a **red wording prerequisite: 5/56
held-out strings fail after the unchanged 9,000 curriculum updates**. Its
fixture now reserves 4,096 percept rows at construction instead of relying
on growth from 64; seeds, updates and assertions are unchanged. After
recording the assertion failure, the diagnostic continues and deliberately
exits nonzero. Ordered prediction beats its controls and related
continuations leave smaller remainders in that imperfect frozen encoder,
but both thought conditions return `(0, 0)` on every question: Brier error
1.0 and mean work 17. The later follow-up below identifies an invalid truth
target in this measurement, so these Brier values cannot establish an
advantage or a null. Alec subsequently required substantial FineWeb training
before evaluating learning quality.

The explicit reconstruction selection passes **4/4**, including three ambient
initializations. The older compose-gradient/cache probe remains unverified
and open: its assertion fails before reaching the cache check.

The final source-matched default sweep completes **4,920 unique cases:
4,595 passed, 324 skipped and one existing expected failure**, with no
unexpected failures. It uses one file and at most 32 cases per fresh worker,
two workers, and the unchanged 8 GiB per-worker cap. Peak worker memory is
7.19 GiB; elapsed time is 1,880.10 seconds. This follows an eight-file worker
that exceeded its cap after 3,731 completed cases. The isolated STM file and
the final sweep both pass without changing assertions, seeds or source.

The global-band probe also passes on CPU and MPS: **6,144 spatial addresses
and 4,096 temporal addresses per device**, with exact recovery. Its CPU
spatial check additionally runs through a full Inductor graph. The slow
wording failure described above remains separate from this passing default
sweep. Changes are uncommitted for Claude's review.

Final documentation-link verification: **96/96 passed**.

### Quality gates require substantial FineWeb training (September 26)

Alec clarified that significant success is not expected from the small,
undertrained examples. The follow-up
makes learning-quality evaluations conditional on a checkpoint with at least
**one million completed FineWeb training sentences**. The exact count and
source manifests are recorded at successful main training updates and survive
checkpoint restoration. Missing evidence skips these evaluations; a qualified
model runs the assertions normally. Small correctness tests remain unconditional.
The old wording curriculum is preserved as a development diagnostic, including
its 5/56 failures; those are not failures of a sufficiently trained model.

The earlier assertion-thought Brier scores above used an invalid target:
observing a statement does not establish its truth. They provide no answer-error
or reasoning-utility evidence. The new test scores an executable next-meaning
forecast after its target arrives. Context-free scoring also now removes role
masks, matching native training; the old context-free numbers are not a valid
clean-control comparison. These corrections do not claim a learning benefit.

With slow tests enabled, the affected selection passes **32 checks** and skips
the **three quality evaluations** for the missing qualified checkpoint.
The default full sweep completes **4,938 unique cases: 4,611 passed, 326 skipped
and one existing expected failure**, with no unexpected failures. It takes
1,408.68 seconds, peaking at 7.19 GiB per worker under the unchanged 8 GiB cap.
The serial seven-update reconstruction baseline remains
.1505906619 / .1347484022 / .1362805218. A real restore regression also fixes
the shape audit rejecting LTM's valid variable-length index column before
its owner can load it. No qualified model's learning performance was measured.
Final documentation-link verification passes **98/98**.
Changes remain uncommitted for Claude's review.

### Occurrence positions and shared-time loss (September 26)

Claude's reduced-period experiment did not remove the loss jump. The new
occurrence/time correction
implements Alec's resulting decision: input percept and symbol occurrences
carry their byte starts, symbol poles reserve one address each, and the
advancing field timestamp stays in the grammar but is excluded from per-word
reconstruction. D3, leaf distillation and both detached reverse layouts apply
that exclusion. General event comparisons can still score temporal differences.

The unchanged seven-update serial measurement gives **.1005906649 before,
.0948241442 during and .0928765051 after training**. At initialization, a zero
reverse prediction against the new sine/cosine timestamp had added exactly
`.1 × .5 = .05` through the time-channel weight. Removing that repeated target
restores the earlier initial cost exactly. The training mean is within 0.1% of
the earlier .0949082300, but final cost is **5.245% above** the earlier
.0882481802. The predicted recovery within 1% did not hold for the final phase;
no seed or budget was changed, and the remaining trajectory difference has not
been isolated. Use the new values as the baseline, not as a learning claim.

Packed and single runs match exactly in their initial parameters and
dictionary, four sentence roots, reference/recovered ideas and byte costs.
Mean byte reconstruction remains **.6839025616645813**. The source-matched
affected selection has **40 passes and 14 opt-in skips**; the explicit slow
selection passes **5/5**, including three ambient native parity runs and the
real D3 position-weight check. The final full sweep completes **4,948 unique
cases: 4,621 passed, 326 skipped and one existing expected failure**, with no
unexpected failures.

The full sweep takes 1442.23 seconds and peaks at 4.81 GiB per worker,
with no compile-cache retries or memory-limit stops. Three workers use one
file and at most 32 cases per batch under the unchanged 8 GiB per-worker cap.

The million-sentence FineWeb learning prerequisites remain unchanged. These
measurements use a tiny training budget and provide no evidence of mature
prediction or reasoning performance. Final documentation-link verification
passes **101/101**. Claude accepted the correction after the green sweep, and
Alec authorized publication.

Implementation `8bc710a` matches all 669 reviewed source blobs.
Claude's review and Alec's publication approval close this implementation.
The mature-checkpoint learning gate remains deferred.
Publication documentation-link verification also passes **101/101**.

### Structural preference and evidence candidate (September 26)

The uncommitted candidate
starts at `074093ec`. Ordinary grammar selection prefers structural operators
on exact learned-score ties; better opaque scores and the existing soft and
sampled distributions are preserved. Boundary measurements count committed
grammar operations and sentences using opaque operations. The measured catalog
contains no opaque candidate: its zero shares do not demonstrate declining use.
The predeclared protocol fixes the
held-out direct-answer and equal-actual-work comparisons across seeds 0/1/2,
retains reconstruction/discrimination and renamed-vocabulary controls, and
reuses the million-sentence FineWeb qualification requirement. Those qualified
comparisons remain pending; learned utility remains unproven.

The unchanged reviewed serial baseline matches exactly at
.10059066489338875 / .09482414424419403 / .09287650510668755. Packed and single
artifacts also match exactly at .6839025616645813 mean byte reconstruction.
The warmed serial observation is .55197 sentences/second with post-batch
routing reads and concurrent CPU work; it is not an isolated speed comparison.

W=6 now preserves every stage and feedback delay in the existing static
scheduler. Both unchanged XOR_grammar CLI gates reach the next failure during
the first epoch's reset/autobind: legacy word/META allocation exhausts the
six-row WholeSpace inventory. The historical unseeded MM loss failure is not
resolved by the new diagnostic. All declared seeds were attempted once: 0 and
2 complete 900 updates below the inherited .20 bar; seed 1 stops at the 8 GiB
guard without a numerical completion record. No seed, capacity or assertion
was substituted or weakened.

The final affected correction passes 108/108 checks, including unconditional
arithmetic poison probes, native-reference/vocabulary renaming and a real
optimizer update. The earlier 207-case affected selection had 204 passes,
two opt-in skips and one retired-context fixture failure; the final fixture
port resolves that failure. Explicit final gates retain two CLI failures and
three quality skips for unavailable qualifying checkpoint evidence.
The single full default sweep completes **4,968 unique cases: 4,641 passed,
326 skipped and one strict XPASS**. It exits 1: the unchanged expected-failure
word-overlap test unexpectedly satisfies its assertion. Its marker and
threshold remain intact, and this unseeded result has not been rerun to change
the outcome. The overlap helper only scores the common position prefix;
the implications need review before making a reconstruction-quality claim.

The full sweep takes 1,419.71 seconds and peaks at 4.82 GiB per worker under
8 GiB per worker / 24 GiB total, with no memory stops or compile-cache retries.
The initial candidate was left uncommitted for Claude's review, with no green
full-sweep claim.

### Structural preference: review corrections (September 26)

The review correction
removes the catalog-dependent scoring-tensor switch: functional selection and
the reconstruction trace both apply structural tie preference to the committed
marginal. The cleared-cache overlap helper requires matching batch/position
counts, so a correct prefix cannot satisfy a per-position reconstruction
criterion. Five deterministic probes fail before the fixes; all eleven new
probe cases then pass. The peer-pipeline fixture, strict expected-failure
marker, overlap threshold and XOR assertions/configurations remain unchanged.

Affected checks complete 140 cases: **124 passed, 15 opt-in skips and one
expected failure**. The single corrected-source full sweep passes **4,980
unique cases: 4,653 passed, 326 skipped and one expected failure**, with no
unexpected failures. The overlap gate scores zero against the original .8
threshold. The sweep takes 1,417.84 seconds and peaks at 7.19 GiB per worker,
under the unchanged 8 GiB per-worker / 24 GiB aggregate limits, with no memory
stops or compile-cache retries.

Serial reconstruction remains exactly .10059066489338875 / .09482414424419403 /
.09287650510668755, and packed/single artifacts exactly match at
.6839025616645813 byte cost. Warmed serial throughput is .49777 sentences/second
with concurrent CPU work and routing reads; no isolated speed claim follows.

The two explicit XOR capacity failures and three missing-qualified-checkpoint
skips remain visible. Learned utility and decreasing opaque routing remain
unproven. These review corrections are validated for the requested commit.

### One operation per round (September 26)

The implementation
replaces compose tiling with one joint tempered softmax over binary operators,
unary operators and eligible STOP. Both hard derivations retain the selected
candidate's probability-weighted straight-through gradient. Explore masks one
uniformly chosen used exploit round, trains, and restores exploit's runtime
records. The tiling DPs, compaction, DP-prior objective, flattened-temperature
pass and separate unary layer are deleted.

Affected validation completes 142 cases: **109 passed and 33 opt-in skips**.
Four explicit compiler/trace checks and 17 ownership/word-store checks pass.
The packed budget/record group passes 20 cases. Later full sweeps expose two
synthetic fixtures requiring the new trace layout; both corrected files pass
all 18 cases. Their changes do not alter production code, configuration or
measurement inputs.

Submission-source coverage completes all **4,945 cases: 4,622 passed, 321
skipped, one failed and one expected failure**. The original pool stopped after
868 completed cases when a relative-STM worker crossed the 8 GiB cap (observed
peak 8.55 GiB). Its three unfinished cases pass in fresh workers at the same
cap; a continuation completes the remaining 4,074 cases. The 868 completed
cases are not repeated. Execution time totals 3,297.45 seconds. All runs
retain the original 1,800-second / 8 GiB worker limits, three-worker / 24 GiB
reservation and total suite deadline.

Serial reconstruction before/during/after training is
.10405020788311958 / .10417700409889222 / .10392807051539421, versus the reviewed
.10059066489338875 / .09482414424419403 / .09287650510668755. Packed/single byte
costs are .787799209356308 / .6835970133543015, versus .6839025616645813 for both
reviewed controls. Initial parameters and dictionary hashes match across the
two layouts, but sentence artifacts and costs do not: exact parity is lost.
These are recorded movements under the existing measurement protocol.

The unchanged MM_grammar gate passes its <.20 assertion; both XOR_grammar CLI
gates fail at the six-row WholeSpace inventory during reset/autobind. The
historical unseeded MM result of .21757 remains visible. The unchanged relative
depth campaign still observes only depth-one end states: STOP is eligible at
depth three, not compulsory. Its assertion is preserved. Mechanism checks and
these short measurements do not establish mature learned utility. Code remains
uncommitted pending Claude review.

## One operation per round: review corrections (September 27, superseded whole-batch draft)

The development draft
uses the shared zero-default sampling temperature, untempered model credit,
raw-logit tie rule, deterministic exploit-only evaluation and forced-prefix
replay under updated weights. This draft retained one whole batch
using the existing scalar objective. Alec subsequently confirmed the amended
per-sentence transaction; the sentence-endings work below
supersedes this draft. Its measurements below remain historical evidence.

Affected tests: **132 passed, 13 skipped**. The focused winner/ownership group
passes all 32 cases, four explicit compiler/packed/graph checks pass, and 109
documentation-link checks pass. The unchanged XOR_grammar CLI gates both fail
at the six-row inventory. MM passes this run; the historical .21757 failure
remains visible. The unchanged depth-3 campaign fails again with all 16 depths
equal to one. No threshold, training budget or gate was changed.

Serial reconstruction before/during/after is
.1040863823145628 / .10699735879898072 / .10230774991214275. Packed/single byte
cost is .7846425250172615 / 1.1408610492944717 with matching initial parameter
and dictionary fingerprints. Parity remains absent; both layouts report
truncation. These are untuned measurements under the existing seed-42 protocol.

Warmed per-batch CPU timings are: exploit/explore forward .359613 / .364350 s,
exploit/explore backward 6.587968 / 6.621047 s, snapshot .025297 s and restore
.001249 s, plus .018701 s for other work. The two backwards account for 94.50%
of the 13.978225-second batch; snapshots/restores account for .19%. This does not
establish long-training utility, and no million-sentence campaign was run.

Nothing has been committed.


## Sentence endings (September 27, reviewed baseline)

Each sentence index
across rows has two optimizer updates from one cached perception: exploit,
then explore under the updated parameters. Reconstruction and prediction stay
per sentence and row; the strictly lower-cost row wins, with exploit winning
ties. Its program and observation commit before the next sentence's perception.
Teacher answers have a separate batch-end backward. Evaluation runs exploit
only; zero-temperature exploration replays the exploit prefix before the
uniformly selected forced alternative. Model credit uses the untempered softmax.

The final targeted selection completes **47 cases: 46 passed, one failed**.
Sentence-closing, causal-context, gradient, temperature, tie, winner, ownership and
reconstruction mechanism checks pass. The remaining output-mode assertion sees
a zero question-conditioner gradient and remains red. The seven explicit checks
complete with **five passes and two unchanged XOR_grammar capacity failures**.
MM passes this run; the historical **.21757** failure remains visible. No quality
threshold, gate source or learning seed was tuned, and no million-sentence
campaign ran.

Reissued serial reconstruction before/during/after training is
**.1065397672355175 / .1001331090927124 / .10013789683580399**. Packed and single
byte cost is exactly **.6496902331709862**, with identical initial parameters,
dictionary fingerprints and per-sentence records. Both layouts still mark every
row truncated. These short measurements do not establish learned utility.

The unchanged two-warm-up/five-measured-batch CPU protocol yields
**.9189261373 sentences/s**, including epoch tails. Measured batch calls average
**2.167488242 seconds**; exploit/explore compose take **1.106395225 / .230206750**,
their backwards **.269424950 / .269619625**, scoring **.111816675**, snapshot
**.000000742** and restore **.000276658** seconds. The first training batch takes
**555.77970 seconds**, chiefly initial capture during scoring. The first measured
batch still takes **5.65698 seconds** and remains in the fixed protocol. Earlier
timings that omitted the sentence reconstruction objective are explicitly
superseded.

The single full sweep completes all **4,968 cases exactly once: 4,636 passed,
321 skipped, ten failed and one expected failure**, exit 1. It takes **5,459.09
seconds**, with **6.19 GiB** peak worker memory and **13.42 GiB** peak aggregate
memory, within the unchanged 8 / 24 GiB limits. No continuation or cache retry
is needed. The ten unwaived failures are the shared-operator reconstruction
gradient diagnostic, the output question-conditioner gradient, four obsolete
evaluation-call expectations, three native observation/prediction-retention
checks, and the unchanged relative-depth campaign. The latter records
**[1, 1, 1, 1]**, with no depth-three state. Both explicit XOR CLI failures and
MM's historical **.21757** result remain visible alongside its current pass.

Work is stopped for Claude's review. Nothing is committed or pushed.

## Reduction pressure (September 27, review stop)

The pressure work
implements the second review's findings and decision 7. The declared default is
`reducePressure=1.0`, adding `d/a + max(0,d-a)/r` to every binary logit, zero
when empty. Online allowance reserves the next word's slot; endings use one
absolute slot or three relative slots. Hard deadlines mask unary and STOP
when every remaining round must reduce. Overflow is an assertion.

Per-closing reconstruction and prediction gradient vectors are captured before
their graphs are released, including the exact cached-perception pullback,
and aggregated into the batch operator report. Four obsolete evaluation call
lists follow the amended exploit-only contract. The observation, conditioner,
operator-gradient and depth-three assertions remain unchanged.

The targeted selection completes **35 cases: 34 passed and the unchanged
depth-three failure**, still `[1, 1, 1, 1]`. All three observation tests, both
output-gradient modes, the operator-gradient report, all 16 interleave tests,
nine sentence-closing tests and three pair-driver tests pass. The explicit
selection has **five passes and two unchanged XOR CLI capacity failures**.
MM passes this run; the historical **.21757** failure remains visible.

The unchanged short reconstruction protocol gives serial before/during/after
values **.11755186505615711 / .10999541729688644 / .09888161532580853**.
Packed/single byte reconstruction matches exactly at **.18002260848879814**,
including initial fingerprints and all sentence records. Every truncation flag
is now false. The fixed two-warm-up/five-measured-batch window yields
**1.5262840033 sentences/s**, including epoch tails. The prior window included
late capture in its first measured batch; the new one does not, so this is not
a clean steady-state architecture speed comparison.

The restored report's overlapping `operator.CS.surface` has reconstruction
norm **1.92338e-13**, output norm **1.73030**, cosine **-.0550583** and
output/reconstruction norm ratio **8.99614e12**. This is measured imbalance,
not evidence that the objectives are balanced or language utility is learned.

The single source-matched full sweep completes **4,988 cases exactly once:
4,663 passed, 321 skipped, three failed and one expected failure**, exit 1,
in **5,631.21 seconds (93.85 minutes)**. All nine failures addressed by the
feedback pass in the full sweep. All 18 new pressure/gradient mechanism cases
also pass. Peak worker / aggregate memory is **6.08 / 13.89 GiB**, within the
unchanged 8 / 24 GiB limits; no retry, continuation or cap stop occurs.

The unchanged depth-three campaign remains red. Two additional fixtures assert
superseded behavior: an externally injected full STM now raises the required
assertion instead of setting `_compose_overflow`, and unary preference no longer
leaves a user sentence incomplete without an LTM row.
The measured and swept source remains intact for the requested review stop.
Both XOR CLI failures and MM's historical result remain visible. No gate, seed
or threshold was tuned, no million-sentence campaign ran, and nothing is committed
or pushed.


## One operation per round: accepted landing (September 27)

Claude's third review and Alec's acceptance authorize the two remaining
fixture patches. The full-STM fixture now expects the admission assertion;
unary preference now expects a completed user sentence and one LTM row.
Production/configuration source is unchanged from the pressure review.

Both complete fixture files finish **60 passed and five skipped**. The one
source-matched full sweep completes **4,989 cases exactly once: 4,666 passed,
321 skipped, one failed and one expected failure**, exit 1, in **5,674.50 seconds
(94.57 minutes)**. Its sole failure is the unchanged depth-three campaign,
still **[1, 1, 1, 1]**. No assertion, threshold or seed is relaxed. Both corrected
fixtures pass in the sweep, as do the observation, conditioner and
shared-operator report regressions. Peak worker / aggregate memory is
**6.16 / 14.19 GiB**, within the unchanged 8 / 24 GiB limits, without retries,
continuations or resource stops.

Serial reconstruction remains the recorded
**.1175518651 / .1099954173 / .0988816153**; packed/single parity is exactly
**.1800226085** with no truncation. Both XOR CLI capacity failures and the
historical MM **.21757** failure remain visible. No million-sentence campaign
ran. Nonzero training temperature and a sentence work term await later specs;
the diagnostic assertion and native context-read flag have recorded follow-ups.

Implementation `6906727` contains all 671 frozen source files. The deferred
compose decisions remain in the task list.

## Two truths, indexed definitions and row-free predicates: accepted landing (September 30)

Claude's review and hand-off, followed by Alec's publication request,
authorize the three remaining fixture ports. A row-free part/implies predicate
primes no inventory row and posts no inventory term; renamed vocabularies
share that predicate while their operands retain independent identities.

The six cases pass, their six complete files pass **59/59**, and all
**202 affected cases pass**, without skips or resource stops. Documentation
links are checked after the final prose. Production code, configurations and
all other tests match the reviewed full sweep. Per the hand-off, no new sweep is run:
the original **4,793 passed, six failed, 322 skipped and one expected failure**
remains intact, and the six test-fixture failures are closed by these checks.
All sixteen failures carried from the two preceding reviews already pass in that sweep.

The sweep took **89.6 minutes**, with unchanged 8 GiB per-worker / 24 GiB
aggregate guards. Graph release passed at **7.631 GiB**. Both XOR_exact CLI
gates pass; both XOR_grammar gates remain red and exact MM_20M_xor round trips
pass **12/15**, with rates .75, .75 and .25 on the three failures. All ten
MM_grammar runs complete, median ending MSE **.1066178977** against HEAD's
**.0696157217**. The older eight-seed reconstruction campaign remains
earlier evidence, with no re-baseline, and the prior depth-three campaign
remains red. The XOR baseline and equal-trial comparison come next;
none of their runtime changes is included in this landing.

Implementation `9810fc7` contains all 712 validated source files and
four supporting fixtures.
The final documentation check passes all 169 links, including todo.md.

## XOR_grammar answer path: stop after five changes (September 30)

The uncommitted work
starts from published `d679df2b`. It implements five changes in order:
the class gate reads the four saved answers at the settled bar; numeric
serial answers read the detached concluded state; grammar word leaves use
the full object code in every binding; code negation reflects the full
code while explicit poles exchange; both sentence trials are scored before
either trains. Each backward owns its perception pullback and prediction
graph. All 35 final affected checks pass.

The mandatory assessment meets the class bar in **0/10 unseeded runs** at
the unchanged 400 epochs, MSE **.2044415629–.5855241919**. The current
reconstruction gate fails 10/10. Per Alec's instruction,
implementation stops before the three remaining changes;
grammar reconstruction, immediate-inverse exclusion and the sum-only
negative control are not implemented or claimed.

All six checkpoints (before changes and after each implemented step)
retain full HEAD/candidate XOR tables, 49 named attempts per tree including
slow proofs and fifteen independent exact round trips. The final table has
**44 passed / 5 failed** on HEAD and **43 passed / 6 failed** on the
candidate: both grammar gates are red, and exact round trips are **12/15
and 11/15**. The historical accepted candidate's 12/15 remains intact.
All ten 900-update MM_grammar runs complete per tree; final median ending
MSE is **.1139670797 / .0686401706** (HEAD/candidate). Starting identical
trees measured **.0647774097 / .0008435599**, showing substantial unseeded
variation. No no-regression claim is made.

The native timing measurement
records final reconstruction MSE before/during/after training
**.1149501931 / .1138415950 / .1189210640**, warmed throughput
**.491403460 sentences/s**, and **5.349 GiB** peak worker memory.
Against the immediately preceding pair, warm batch time is **1.4410×**
and whole-run peak memory **1.0111×**, with all late graph captures kept.
These are different unseeded initializations and concurrent workloads,
not paired causal measurements. The historical parity fixture fails to
load on both HEAD and candidate because its `propertyBasis` element was
retired; no configuration is changed to bypass it. The depth-three check
skips without the mature FineWeb checkpoint, leaving the historical red
campaign and its assertion intact.

The final source-matched full sweep
completes all **5,161 cases exactly once: 4,832 passed, six failed,
322 skipped, one expected failure**, in 5,985.72 seconds. There are no
missing cases, resource stops or compile cache retries. Peak worker and
aggregate memory are **5.002 / 17.909 GiB**, under the unchanged 8/24 GiB
guards. All 12 added regression cases and 27 added documentation cases
pass; no previous selector was removed.

The six unresolved failures cover two real full-graph numeric-head paths,
one unaligned object-reference assertion, definition identity, and two
truth-ingestion evidence assertions. The full-graph head reads concluded
state through an attribute unavailable during tracing instead of receiving
the explicit sentence products. These failures and their unchanged
assertions are retained for review after the mandatory stop. The earlier
sweep's six fixture failures were already ported in the accepted September 30 landing;
their passes are not attributed to this candidate.

No seed, optimizer, epoch budget or other configuration was changed. The
candidate stops for Claude's review without a commit.

## XOR_grammar repairs and isolated measurements (October 1)

Alec and Claude's October 1 instructions replace repeated HEAD/candidate
measurements with one candidate attempt per named case, including one
MM_20M_xor exact round trip. The accepted September 30 record remains the baseline:
XOR **44/49**, exact **12/15**, MM_grammar median ending MSE **.1066**.
No test body or assertion changes in this follow-up.

The compiled answer head now receives the concluded state explicitly.
The mixing binding stages object leaves for grammar without publishing
word-symbol rows, IDs or evidence; this also repairs the definition identity
alias and both stored-truth evidence failures. All **12 affected checks
pass**, including all six previously failing cases and the object-leaf and
answer-gradient boundary checks. The candidate XOR table is **33/35**;
both grammar gates remain red and the single exact round trip passes.
All named slow proofs are included. This sample does not replace the
historical exact-round-trip rate.

The three requested probe patches are installed only in measurement
processes; the candidate remains at its fifth change with its answer gradient cut.
At 400 epochs with all four classes correct and MSE below .05, ten fresh
unseeded runs meet the bar in **0/10** for detached answer comparison,
**0/10** with four explore trials, and **9/10** with the answer gradient
reaching codes and chooser. Median MSE is **.2382143314**, **.2939883538**
and **.0010992477** respectively. Every trial is costed before training any
trial, under identical parameters. The four-explore probe trains all five
trials, so its additional optimizer steps accompany its wider search.

The final MM_grammar measurement
contains ten unseeded 900-update runs. Median ending MSE is
**6.575284761e-11**, with range **2.220446049e-16–.25**.
The helper matches the accepted September 30 landing byte for byte and uses the
same `eager` compilation setting. The environment was not rebuilt:
before/after `pip freeze` matches exactly.

Two harness issues are disclosed: a two-slot derivation
observer needed the full STM, and the initial MM dispatch inherited the
variant compilation setting. No completed statistical failure was
retried. The measurement patches and candidate source are unchanged by
these harness repairs. The next part of the plan has not started; no new full sweep or commit
is made before review.



*From `Testing.md` lines 3053-3133, moved 2026-10-30.*

## October 3 decoder exploration, operators and one attention — review candidate

The candidate starts from `802abb1a` in the existing working tree and remains
uncommitted for Claude's review. The final default suite completes all **5,115**
selected cases: **4,829 passed, 285 skipped, 1 existing xfailed**, exit 0, in
123.65 seconds. Aggregate peak is 14.03 GiB; maximum worker peak is 3.74 GiB.
The 8 GiB / 1,800-second worker guard, 28 GiB aggregate cap, ten-worker maximum
and CPU-headroom policy remain unchanged. This is not a replacement for the
previous failed weekly slow run.

The unchanged MM_xor acceptance test passes at **.1826079786** within its
200-epoch allowance (49 observed loss evaluations). All twelve grounded and
word-boundary XOR variants pass at exact zero, and both original XOR_exact CLI
checks produce **MSE 0 and reconstruction 4/4**. Forty-six retained reading/global
capability checks pass, and the four actual shipped reference configurations
pass finite forward and answer-reader gradient checks. The nine WholeSpace row
increases required for the fixed word whole are disclosed.

**Held for the ten-run measurement, not rejected.** The candidate's single shared 400-epoch training
records MSE **.3207185981**, three correct class labels and reconstruction **0/4**,
with **zero ownership conflicts** over 1,200 backwards. The accepted XOR_grammar baseline
remains **.1147481948, 0/4, zero conflicts**, with prior class **9/10** and
reconstruction **5/10** unchanged. All five completed source-stage measurements,
including the earlier bad operator-stage result, remain visible. None is a
paired-initialization campaign. The candidate's single runs span .0013 to .875;
the .321/.115 comparison is not a regression finding. The requested ten-run
comparison uses one training per run for both unchanged bars against 9/10 and
5/10, with ten MM_xor acceptance runs and ten sum controls.
The decoder explores strictly better reconstructions but its greedy operation
inference still stops at one leaf.

Alec clarified that the NanoChat gate waits for a trained checkpoint.
Fresh BasicModel scoring stopped. One small evaluator mechanism check passes on
`MM_grammar_wording.xml` over the first three unchanged manifest items, all 16
choices and both prefix controls. No partial result is a language gate.

Operators and
one-attention mechanisms are implemented as a review candidate. The October 4 review
requires sampled departures, the decoder's logit-margin/gradient audit and
frozen evaluation without definition admission before the ten-run measurement.
Nothing is committed until Claude reviews those results.

**October 4 measurement, completed once on frozen source:** XOR_grammar
class **7/10**, reconstruction **0/10**, joint **0/10**; MM_xor **10/10**;
sum control **0/10**. Each XOR training supplied both unchanged original bars.
XOR final-error bands are **7 at 0, 0 at ¼, 1 between, 2 above ¼**. The
class/reconstruction/sum counts are below the prior 9/10, 5/10 and 10/10.
The audited tenth run records zero ownership conflicts over 1,200 backwards,
decoder kept-path stability **.610763**, and nonzero gradient-driven margin motion,
while STOP stays ahead of both binary undo choices throughout the budget.
The targeted checks pass **129 tests with one existing skip**, and the
one-batch observer check passes. No new full sweep or native run was added.

**October 4 measurement, completed once on frozen model source:** the
numeric reader is affine and decoder eligibility follows the disclosed
single-code/pair residual comparison. XOR_grammar class **2/10**,
reconstruction **6/10**, joint **2/10**; MM_xor **10/10**; sum control
**10/10**. Bands: **2 at 0, 5 at ¼, 3 between, 0 above ¼**. Class is below
the 9/10 comparison; reconstruction is above 5/10. No measured model was
repaired afterward. The tenth run has zero ownership conflicts and decoder kept-path
stability **.339487**. STOP is ineligible at all audited first steps and the
fixed-parent policy-margin change is zero.

The focused checks plus the reader and echoic fixture ports cover **139
distinct passing checks**, and the one-batch audit check passes. Two
test-only fixture ports followed measurement. Compiled decoder checks pass;
one optional full-model compile probe was interrupted before measurement and
is not counted as passing. No full sweep or attribution training was added.
Fresh BasicModel scoring remains stopped, with the trained-checkpoint gate pending.
Nothing is committed; the result is held for Claude's review.


*From `Testing.md` lines 3134-3199, moved 2026-10-30.*

## Forgetting candidate (2026-10-19)

The initial candidate default sweep
completed **5,805 cases: 5,470 passed, 334 skipped, one expected failure,
zero failures**, in 464.38 seconds. Peak worker memory was 4.58 GiB under
the unchanged 8 GiB limit. XML schema validation and `git diff --check` pass.

The 44 forgetting-specific cases cover per-stream age, source trust distinct
from both evidence poles, one-step proofs, protected/live references, gradual
coarsening, expectation unlinking, taxonomy provenance, slot reuse without
identity reuse, optimizer moments, relevance pruning and checkpoint state.
The focused packed/single and existing nested/expectation retention selection
passed 113 cases before the final protected-metadata fix; the final full sweep
includes that fix and its regression.

The cleared-cache word-overlap test remains expected-failing. Copular identity
(1), FineWeb (5) and stage 4e (1) retain their phase prerequisite skips.
This work starts no separate NP, identity or corpus learning experiment and
makes no mature-checkpoint learning claim. Claude reviewed that candidate;
the repair pass below supersedes its source and is approved for landing.

### Forgetting review repair pass (2026-10-19)

Stored surprise now governs ordinary
three-role sentence rows as well as bare ideas. Unused mints without
structural or reference evidence can be reclaimed independently, smallest
magnitude first; defined and referenced concepts remain protected from
smallness-based reclamation.

The real BasicModel certificate logs the first document's **11→6** LTM
pass at capacity 16, then reads and writes a second document. Its post-pass
checkpoint restores occupancy, addresses and all four forgetting counters
exactly. Focused checks had **93 passed, 2 existing skips**; the final isolated-source
forgetting/concept selection passed **47**. XML and whitespace checks pass.
No existing assertion is relaxed, no test seed is selected, and no separate
learning experiment is run.

The single default sweep on **b1a19c5b8ec94941d09f74190913f149d42484c1**
completed **5,808 cases: 5,471 passed, 334 skipped, one existing XFAIL and
two failures**, in **472.34 s** (exit 1). All three new repair certificates
and both registered artifact-restoration certificates passed. The source
remained unchanged throughout; peak worker/aggregate memory was
**4.80/16.29 GiB**, below **8/28 GiB** limits. There were no compile-cache
retries. Phase skips are unchanged: copular identity 1, FineWeb 5, stage 4e 1.
The isolated checkout reported no weekly slow-test record; none was run.

| Default-sweep failure | Classification and disposition |
|---|---|
| `test_relative_links_resolve[doc/Params.md]` | **Environmental:** the isolated checkout lacked the existing parent repository's `doc/Ethics.md`. The companion path is now present; source and assertion are unchanged. The scoped recheck is recorded below. |
| `test_both_training_trials_share_bootstrap_snapshot` | **Carried finding:** **0.0517** where `audit['components'][..., 0]` must be zero. Claude identifies the same paired zero-cost counterexample reproduced on accepted main on October 15, with the sampled row varying across runs. Alec carried it to the full training session on October 17, outside this landing's conditions. The assertion is unchanged and captured output has no forgetting report. No rerun was used to obtain a pass. |

Claude and Alec approved landing with these classifications. The subsequent
approved cleanup uses utility 1 directly in the proof-loss raise, deleting the unreachable
`REL_NONE` surprise branch because `_supports` only keys part/implication
rows. Alec explicitly requested **no re-sweep**; none was run.

The scoped link recheck passed **4/4** after restoring the companion path. The
default result remains unchanged, and the paired zero-cost counterexample
remains carried to the full training session.


*From `Testing.md` lines 3200-3246, moved 2026-10-30.*

## Carried invariants and gates: second repair (2026-10-19)

The landing source is **`bfc834d79`**, resolving the reviewed `c4e3520a2`
mechanism failures. The production prime
is on and uncapped. Under parity, certificate (b) measures the existing
chooser's own content score with form and expectation held out: 5,204
menus and 3,864 retained alternatives lie at **1.0–1.000000030** of
uniform, inside the unchanged [.9, 1.1] band. Known antecedents train the
existing form/content terms by marginal likelihood.

At the original one-successor initialization, **4199722147**, sentence A
had compared whole-question surface reconstruction before the episode.
The component diagnosis
records the old 1.92 versus 1.05 and the repaired binding judge. After
the proof, the correct binding costs **0**, versus **.66483945** for the
wrong departure. An open reference whose menu contains only self-inverse
negation can conclude without pretending it found an antecedent.

Both unchanged decomposition certificates pass in both gain arms,
with constructor parameters matched to the original runs. All eight
companion questions have the required bindings and witnessed chains.
The complete scoped selection finishes **47 passed, one copular
prerequisite skip**. The final boundary selection passes **475/475**,
including the original paired reconstruction certificate and all five
compiled-boundary cases.

The single standard landing sweep completes **5,861 cases: 5,540 passed,
322 skipped, one existing XFAIL, one failure**, in **1,353.24 seconds**.
The paired reconstruction and common-constructor PerceptMoments cases
pass. Peak worker/aggregate memory is **2.88/7.55 GiB**, within 8/28 GiB.
The empty-menu failure exposed that the new `not advances` condition was
also true when there were no operations. Its repair requires a nonempty
menu, preserving the original `no_legal_operation` finish. Scoped
controller/reference checks and the complete file finish **113 passed,
one copular skip** after that repair. Both unchanged decomposition
certificates pass again in both original arms (**4/4**); all eight companion
questions retain their correct bindings, witnessed chains and work counts.
The final certificate (b) observation has 3,320 menus and 3,864 retained
alternatives in the same **1.0–1.000000030** band. There is no second full
sweep. The copular
producing lesson remains after stage 4e in the mechanism residue. Graded/shuffled,
derivation-optimality, long-run collapse and the separate learning
track remain open; no failed learning run is resumed or regraded.

# GradientFlow.md

*From `GradientFlow.md` lines 547-587, moved 2026-10-30.*

The eager training audit additionally records the first decoder step's raw
logits and `STOP − undo` margin for each binary action, on both decoder paths
within each compose trial. Hooks record the gradient reaching those logits
only during the actual owned backward; diagnostic gradient queries do not
count as training. Each optimizer step records those gradients, their
`dL/dSTOP − dL/dundo` difference, and the margin before and after the update
with the same detached parent held fixed. This distinguishes policy motion
from changes to its input representation. Steps that do not reach decoder
logits are recorded with no walks. No RNG draw, extra backward or optimizer
update is added. Raw gradients and actual margin changes must both be read:
shared policy parameters and momentum can move a margin differently from
independent descent on its two logits.

The saved campaign measures
class **2/10**, reconstruction **6/10**, joint **2/10**, MM_xor **10/10** and
sum control **10/10**, once on the frozen model source. The class count is
below 9/10 and remains for review. On the tenth run, ownership conflicts
are zero (23 active, 64 inactive parameters over 1,200 backwards). Every
audited first-step row/path masks STOP as compound. Its raw margin remains
near 2, but its gradient is zero; undo-gradient differences are at most
1.96e-9 and all fixed-parent margin changes are exactly zero. Changing raw
epoch means reflect changing parents, not an observed policy update.
Decoder kept-path stability is **.339487**, with zero explore wins and zero
strict-rule violations. The audit supplies these observations without assigning
a cause to the class failures. The candidate is uncommitted for Claude's review.

The composition-mechanism campaign
measures class **0/10**, reconstruction **7/10**,
joint **0/10**, MM_xor **10/10**, and sum **10/10**.
All ten sum controls were read before the gates started. Comparison is the
saved campaign (2/10, 6/10, 2/10, 10/10, 10/10); below-comparison counts:
**class_pass 0/10 versus 2/10, joint 0/10 versus 2/10**. The final run records **0 ownership conflicts**
and decoder kept-path stability **0.009387**. Its first-step eligibility
counts are `{"compound": 6400}`, and the largest absolute
fixed-parent margin change is **0.000302553177**. No attribution training or retry was added.
The saved campaign remains as measured: only its tenth run saved final compose operators,
all mean disjunction; the other runs' operators cannot be recovered from their
scores. The five-run mean hypothesis is unconfirmed. This candidate remains
uncommitted for Claude's review.

*From `GradientFlow.md` lines 743-759, moved 2026-10-30.*

## Expectation at the closing

At the sentence closing, `ClosingImage` forms detached $n = -g \cdot (1-m) \odot \kappa \odot \hat{e}$
and `c = o+n` on the concept complement only. The complete perceptual block,
including its reserved coordinates, is protected. Thought reads this closing's
`c`; storage and predictor targets retain `o`. The retained image restores
`o = c−n` at float precision. This supersedes the ad-hoc image formerly
constructed in the thought context. Prediction still minimizes its unchanged
all-role relative squared error and presence/kind BCE against detached targets;
the confidence gate changes comparison costs, not predictor training. Empty
roles have zero targets, not zero residuals. There is no new loss or owner.
At the credit landing (`3de37eef5`), XOR_grammar and MM_grammar had zero complement coordinates (22 total,
22 in the perceptual block, of which 14 carry form content). The October 7 work adds
bipolar meanings; these gates still have a zero image because no prior
forecast is available. The predictor takes only the committed row as target.



*From `GradientFlow.md` lines 852-899, moved 2026-10-30.*

## Score-function attention, exact pair recomposition and the teacher-forced walk (October 5)

**Attention uses the same estimator as compose.** The input walk samples one
eligible round uniformly and one non-greedy legal action uniformly. With K
alternatives and R eligible rounds, its surrogate is
`K·R·p(a_departure)·detach(C_explore−C_greedy)`. Both native-percept byte costs
are measured before any owner step, with the greedy cost as baseline. Features,
keys and priming are detached; hard attention values have no straight-through
credit. The term is consumed once at reconstruction's first sentence owner
step (or the batch owner step for a field-only path). Exact ties attach no
gradient or momentum-only update. Compose retains
its own sentence comparison; both terms have the same reconstruction owner.
The tenth-run audit records costs, actions, K and R, advantages, probabilities
before/after the actual update, analytic gradients and finite differences.

**Exact pair recomposition precedes learned context.** The bounded shortlist
and candidate limits are unchanged. A squared relative residual at most
`(8·finfo(dtype).eps)²` is float-roundoff exact; if one exists, the residual
argmin wins independently of all context weights. Equal residuals retain bank
order. The pair scorer's CE still sees every eligible pair. Each operand's
projection coefficients are centered and divided by their population standard
deviation over its valid shortlist; constant features become zero and invalid
slots contribute neither mean nor variance. Codes and these features are
detached. Missing targets are still counted without a CE term.

**The walk policy is teacher-forced.** Each actual compose parent teaches its
declared undo or unary generate face; each input leaf teaches STOP. The states
are detached saved values. Cross-entropy averages nodes within a sentence and
then active rows; it is added as `reconstruction.walk_policy` after the paired
byte-cost comparison, alongside `reconstruction.decomposition`. No imitation
loss enters the forward chooser or trial selection. At inference the policy
chooses freely under the existing eligibility mask, with no compose journal.
Its straight-through transition blend is removed. The free decoder logits
receive no gradients.

In the tenth measured run, all 1,600 attention advantages and all
1,600 compose advantages were zero. Those records confirm tied-cost behavior,
not a nonzero score-function learning signal; the positive/negative-advantage
analytic and finite-difference checks are separate focused tests. The 3,200
walk-teacher records match CE gradients within 7.45e−9 and show actual policy
logit updates. Code displacement, sentence-path perception gradients and
ownership conflicts remain zero. The measurement originally missed the standing MM_xor count (9/10). Alec
accepted this work on October 6 after an audit established that its
MM trajectories were identical to the landing. The landing is `73cd7b71b`.


*From `GradientFlow.md` lines 900-943, moved 2026-10-30.*

## The target scheme (October 6; decided)

Decided, with the architecture statement in
[Architecture](Architecture.md#expectation-and-surprise-through-the-architecture).
The sections above describe what is implemented; this one what each step
changes in the gradient's ownership and path.

- **The trial cost (October 6, corrected twice that day).** The chooser is credited by
  the owner-step total `R + E + A`; only reconstruction `R` decides the keep.
  *(Superseded 2026-10-19: keep and credit both read `R + A`; `E` trains the
  predictor only.)*
  Expectation is gated per role by $g \cdot \kappa$; the supplied answer is read on both
  detached roots before either step. The reader takes one mean-loss step on
  both compose roots, or one step on the kept root for a narrowing departure.
  Strictly lower reconstruction keeps explore; a tie keeps greedy. The SCG
  surrogate, reduction, registration and ownership remain unchanged.
- **One departure over both walks (October 6).** Narrowing and compose share the
  sentence's single departure (`R` over both walks' eligible rounds); a
  narrowing departure runs through the sentence; the percept-stage selection
  and the separate attention surrogate retire. The walk's poles are handed off
  with the scope, so a field departure can change a cost.
- **The gain at the closing (October 19).** `c = m*a*o`, with one
  scalar per native identity. The raw observation remains stored. Independent
  excess and omission lanes train the local predictor and set surprise. At
  zero gain, restoration reads retained `o`, never divides by the gain.
- **Identity by construction (October 7).** Pairs plus length as the parts; the
  content width for a sparse superposition; the narrowing-weighted ceiling
  and the centroid. Reconstruction of forms becomes exact by construction.
- **Expectation across the model.** One shared activation predictor
  is read at S, R and D, from detached observed activations and STM. P uses
  running mean and variance. Prediction by inversion and the broader
  invertible-path audit are deferred; they do not make E's sources live.
- **Attention is a mask over `.where`** (Alec, 2026-10-07), not a relevance
  filter: the narrowing walk from the sentence's global scope to each word,
  every word visited. Its loss is what the mask leaves out of the target,
  charged by the owner-step trial cost and credited to the walk's choices by
  the score-function term; the region inside the mask pays the
  percept/concept chain's cost. No per-word relevance weight is trained.
- **Unchanged throughout.** Perception's codes have one writer; the forward is
  a tree; the committed root is what the readers see at test and what the
  class bar measures; no straight-through, no mixed forward.


*From `GradientFlow.md` lines 944-992, moved 2026-10-30.*

<a id="credit-october-6-uncommitted-candidate"></a>

## Credit: one departure and the owner-step trial cost (October 6; rejected candidate)

The candidate starts from the
accepted October 5 source. Sentence comparison uses the Error registry's relative
`R+E+A`, with all three components and their signed differences retained for
review. Presence and role errors are multiplied by detached $g \cdot \kappa$ in the
numerator; their uninformed baseline is unchanged, so zero confidence cannot
cancel out of the ratio. Kind error uses the mean role gate. Predictor training
keeps its original ungated registry. The supplied answer is a detached read of
each complete trial, and reader updates remain restricted to the kept rows.

One uniform departure ranges over narrowing and compose. Its sole surrogate
is `K·R·p(a_departure)·(C_explore−C_greedy)`, registered under the existing
`reconstruction.compose_score_function` name and reduced over active rows.
Exact ties disconnect the policy term. Percept-stage byte selection and the
separate attention surrogate and consumer are retired. Packed sentences use
their own eligible rounds and retain the shared input work allowance.

Narrowing hands off each operated bracket's bilattice pole pair to the words
inside it. Unoperated words keep native evidence; a native descent supplies
its identification witness. Scope, admission and poles belong to the trial.
The operated extents survive splitting; their field operations resolve in
walk order over the completed native witnesses, including words identified
by a later descent. An earlier unknown pair cannot erase that identification.
The sentence consumes the explicit pair where declared, or its detached net
evidence as leaf activation, and the closing carries its polarity. No field
value is handed off, and no stored code or binding kernel implements negation.
The closing image described above is the other mechanism in this candidate.

Pair search, exact-fit precedence, teacher-forced decomposition, reconstruction
objectives, owner lists, budgets and the bands are unchanged. Focused
checks cover both departure gradients, total-cost selection, pole evidence,
the image table and restoration, and the committed-root answer.
Implementation alone does not establish the standing gate.

The frozen campaign completed all thirty unseeded trainings without
retry. It **missed the standing gate**: class 4/10, reconstruction 6/10,
MM_xor convergence 9/10, sum 10/10; the full sweep is green and the ownership
zeros are retained. Across XOR, 5,426 of 16,000 sentence records have nonzero
advantage, including 1,074 narrowing departures; both walks' logits move.
The initial observer omitted both reference-slab consumers and therefore
incorrectly treated MM as inactive. The first forward differs from the landing
at identical construction parameters and RNG state. MM's 9/10 is a live miss.
The candidate was not accepted.



*From `GradientFlow.md` lines 993-1036, moved 2026-10-30.*

## Keep and credit separated; the sign-only pole handoff (October 6; rejected candidate)

The keep
and the policy's credit are distinct: strict improvement in `R` commits explore,
while `Δ(R+E+A)` supplies the unchanged `K·R·p(a)` score-function surrogate.
The audit records both components per trial, the reconstruction decision,
the advantage sign, and whether the answer reverses credit against the keep.
Both detached roots train the answer-owned reader on every active supplied row.
The single departure, predictor training, reconstruction teachers and owner
lists are unchanged.

The handoff changes only activation sign, retaining its magnitude and the
leaf event exactly. A tied pair retains the prior sign. Explicit-pole grammar
reads the full pair. Untouched scalar references use the landing's exact
signed-activation conversion, including saturation to `[0,1]`; reached fields
choose its expressed pole. `ModelAttention.POLE_CONSUMERS` declares and the
observer counts the sentence payload, explicit-pole pushed slab, and both
reference-slab publication paths. The closing reads the resulting evidence.
The concept-only image is unchanged and remains zero in the two grammar
configurations whose complement width is zero.

The pre-training disjunction replay exposes a remaining downstream form
change: `InterpretLayer.forward` multiplies the signed activation into the
resolved object code. With negative poles, the unchanged disjunction receives
negative directions, whose composition is not the negative of the positive
composition. Merely removing handoff rescaling does not repair this inverse.

The bank-wide inherited-part audit is read-only and covers order-zero forms.
It checks all distinct row pairs whose positive-net part sets are included,
using intersections of the part postings. It reports pair counts and maximum
coordinatewise violations before and after supplied placement snapshots;
current codes are `L` on both sides. The conceptual complement is unconstrained.
No centroid placement or higher-order enforcement is added in this candidate.

The campaign completed all thirty unseeded trainings: class 1/10, reconstruction 6/10, MM 10/10, sum 10/10. Its standing gate failed; the sweep is green. All three paired MM trajectories differ from the landing at the first forward, so MM is live. The candidate remains uncommitted for review.

Forward-only tracing distinguishes the MM configurations: `MM_xor` bypasses
all four declared pole consumers, while `MM_grammar` reaches both reference
paths. The paired MM_xor trajectory difference is established independently;
its causal attribution to the reference slab is not supported by that trace.



*From `GradientFlow.md` lines 1037-1086, moved 2026-10-30.*

## Form and meaning signs; one reader update per sentence (October 6; not accepted)

Interpretation
and retained leaf values apply activation magnitude to the form block, signed
activation to the meaning complement, and leave occurrence coordinates alone.
The binding kernel receives forms without a negation sign. The explicit pole
pair and closing polarity still express negation. Reference publication uses
the resolved grammar rows as the word brick does; an absent serial row must
not erase a resolved leaf's evidence. Both publication paths use this rule.

The reader receives one optimizer update per supplied sentence. For a compose
departure its objective is the mean of the two detached-root losses; for a
narrowing departure it is the kept root's loss. Mixed batches weight each row
separately. Both reads precede either trial update; the first reconstruction
step omits the reader, and the second carries the combined reader gradient.
The batch-end answer remains observable without training the reader again.
Actual reader Adam counters are recorded alongside the loss weights.

Reconstruction alone keeps a trial, strictly; `Δ(R+E+A)` still credits the
single departure under the existing SCG registration. Predictor and
reconstruction updates, owner lists, the image and containment audit remain.
In the two grammar configurations the meaning complement is empty: a forced
narrowing `not` flips evidence and closing polarity but leaves the root and
all comparison costs identical. A nonzero-complement fixture checks that only
meaning changes sign. The saved negative-pole disjunction fixture recovers
four multisets with zero pair-argmin residual and exactly the actual one-attention
landing's owner reconstruction cost at the same saved initialization, before
any gate training.

MM_xor's changed first forward is caused by RNG consumption, not a pole
consumer. The retired percept trial draws advance the generator used later
by `create_ir_mask`'s Bernoulli input mask. At seed zero its byte costs tie,
its kept walk and scope match, and restoring the old trial restores the
landing output. Restoring the RNG to the state after the greedy walk restores
the credit candidate's output. MM remains a live gate, with three paired trajectory
replays recorded separately from the thirty unseeded trainings.

The thirty trainings produce class **5/10**, reconstruction **10/10**, sum
**10/10 at the quarter floor**, and live MM_xor **9/10**. The gate as read at
that measurement was missed on class and MM; it subsequently excludes the
bisection-proven RNG-only MM path. Every final grammar root uses conjunction; the five
class misses remain above the MSE bar. All 10,650 XOR and 16,000 sum
narrowing departures tie. All 5,350 XOR compose departures have nonzero
credit; each grammar run's reader Adam counters end at 400 after 400 epochs.
The full sweep is green (4,971 passed, 285 skipped, one xpassed), and perception
gradient, code displacement and ownership conflicts remain zero. All runs
and failures are retained without retry. This candidate is held for Claude's
review and is not committed.


*From `GradientFlow.md` lines 1087-1131, moved 2026-10-30.*

## The departure proposal over both walks and its correction (October 6; review candidate)

The only
runtime change is the sentence departure's proposal and correction. Eligibility
is computed on both greedy walks for this sentence. A uniform draw among the
nonempty walks precedes a uniform eligible-round draw within the selected
walk. The existing uniform alternative draw then chooses the departure action.
One available walk gives `W=1`; no available walk gives no departure. The
surrogate multiplies by `K·R_walk·W`, canceling that proposal exactly. Its
active-row reduction, greedy baseline and reconstruction registration remain.

The strict reconstruction keep, `Δ(R+E+A)` advantage, one reader step (mean
detached roots for compose; kept root for narrowing), magnitude interpretation,
owners, budgets, image and containment mechanisms remain as in the preceding candidate.
Every observed sentence records its walk, `W`, `R_walk` and `K`; both walks
have analytic and finite-difference checks. The focused proposal check gives
equal walk frequencies despite unequal numbers of eligible rounds.

A path changed only by global RNG consumption,
established by bisection, is not a regression gate. The seed-zero MM bisection
is repeated on this candidate; raw counts and paired trajectories are still
reported. The thirty gate trainings remain unseeded, without retries or
replacement runs. This candidate is held for Claude's review before a commit.

Measured: class **2/10**, reconstruction **10/10**, sum **10/10 at ¼**,
and raw MM_xor **10/10**. The standing class gate is missed. All forty final
grammar roots are conjunction and all labels are correct; eight runs remain
above the MSE bar. The three all-disjunction starts become all-conjunction
from epochs 22, 65 and 106, and three failed runs already use conjunction
throughout. Faster selection has not recovered the class count.

There are **7,978 compose** and **8,022 narrowing** XOR departures; every
narrowing advantage is exactly zero, as are all 16,000 sum advantages.
Every recorded correction equals `K·R_walk·W`. All compose advantages are
nonzero; the answer credits against reconstruction's keep on 1,375 rows.
Every sum/XOR run has 400 reader updates and final reader Adam counters of
400. Across all recorded sentence rows, maximum analytic and finite-difference
discrepancies are `2.23517418e-8` and `1.76165817e-8`; focused checks also cover
a nonzero narrowing advantage. Chooser logit ranges move in every XOR run.
The full sweep is green, with zero sentence-path perception gradients,
code displacement and ownership conflicts. The repeated MM bisection confirms
the RNG-only path and all three paired trajectories differ. No retry or
replacement changes the count. The candidate remains uncommitted for review.


*From `GradientFlow.md` lines 1132-1174, moved 2026-10-30.*

## Presented and comparison readers (October 6; review candidate)

The candidate separates two uses of the answer read. The presented reader takes
one step on the reconstruction-kept root alone. The comparison reader has
the same form and independent answer-owned parameters; it retains the preceding candidate's one
step on the mean of both root losses for compose departures, or the kept
root for narrowing departures. Only the comparison reader's answer cost
enters `Δ(R+E+A)`. Presentation, writing and the class gate use the presented
reader. Its weight on the rejected root is exactly zero.

The comparison weights are cloned before the first update without consuming
RNG. Both roots remain detached. Each reader's independently reduced loss
enters the existing answer objective, so each takes one update without an
extra batch-end step. The strict reconstruction keep, walk-first draw,
`K·R_walk·W` correction, other components, registration, reduction and owner
boundaries remain as in the preceding candidate. The frozen AST comparison verifies the unchanged
reader computation and credit functions; focused tests verify separate
gradients, tied-parameter restoration, counters and public-output isolation.

Each run records both readers' MSE before the trial updates on the same kept
roots, their active weight norms and the stable greedy conjunction epoch.
Final class evaluation remains the ordinary presented read after training.
The MM seed-zero bisection is repeated and the paired diagnostics remain separate
from the thirty unseeded gate trainings. There is no retry or replacement.

Measured: class **9/10**, reconstruction
**10/10**, sum **10/10 at ¼**, and raw MM_xor
**10/10** under the repeated RNG-only bisection. The measured
standing gate is met; acceptance remains pending Claude's review.
Final evaluation has 40/40 conjunction roots and 40/40 correct labels.
Every run uses greedy conjunction throughout from epoch 140 at the latest.
All twenty sum/XOR runs have
400 updates per reader, zero presented weight on the rejected root, and
final active reader Adam counters of 400. The complete source sweep is
green; ownership conflicts, sentence-path perception gradients and
code displacement are zero. All thirty unseeded gate trainings are
retained without retries or replacements. No commit or push.


*From `GradientFlow.md` lines 1175-1234, moved 2026-10-30.*

## Identity by construction: the fixed word form (October 7; review candidate)

The native word form is now fixed by boundary pairs, cumulative length
parts and recorded collision mints. The pair block has 64 coordinates;
the separate 32-coordinate length block is a thermometer. Fixed rows are
read independently of optimizer weight decay or updates to the learned
byte rows. Atom codes and the dense projection use private generators, so
admission consumes no global training RNG. The projection is a buffer,
not an optimizer parameter.

The sparse form addresses the native index and supplies the containment
and exact rung-zero byte audits. Binding and pair search read its fixed
96-by-64 dense projection; activation still supplies magnitude. The
projection changes code directions, not the conjunction or disjunction
kernels. Resolved live conceptual references already have their binding
directions and are not projected again. Journal leaves retain the directions
used by the operation that produced their root.

An identified word's byte reconstruction cost is exactly zero. This term
audits identity and does not train a byte decoder or the fixed form. Other
reconstruction terms remain in R, and an R tie keeps the greedy trial.
The presented reader still trains on the kept root; the comparison reader
still trains on both compose trials or the kept narrowing trial. Their
detached roots, independent answer ownership, walk-first proposal,
K·R_walk·W correction and registration/reduction remain as in the two-reader candidate.

Signed ingestion trust now sets the row's positive/negative pair; negation
exchanges its poles. A derived signed difference replaces the independent
provenance column. This changes ingestion evidence, not the binding kernels.

The thirty unseeded gate trainings and MM seed-zero bisection
remain separate; no retry or replacement changes a gate count.

Measured on the frozen candidate: class 10/10, reconstruction 10/10,
sum 10/10 at ¼ and raw MM 10/10 under the RNG-only amendment. All forty
final XOR roots are conjunction with all forty labels correct; every run
is permanently all-conjunction by epoch 41. All 32,000 sum/XOR comparisons
tie on R and keep greedy. The 8,040 compose departures carry answer credit;
all narrowing departures have zero advantage. The presented reader's
rejected-root weight, ownership conflicts, sentence-path perception
gradients and code displacement remain zero. The rung-zero audit reports
256,000 identified reads with zero errors. The complete frozen sweep has
5,003 passes, 285 skips, one XPASS and no failures across 5,289 cases.

Both operators' projected roots are affinely separable. The saved centered
singular values are (0.877855, 0.800607, 0.482921) for conjunction and
(0.691670, 0.523711, 0.0588575) for disjunction; their nonzero condition
numbers are 1.82 and 11.75. Disjunction requires a larger affine read but
its separating direction is well above rounding noise. These spectra
support a convergence difficulty difference without proving the cause of
the comparison reader's preference. Three passing presented readers retain
late common-offset oscillations; their roots are identical to the other
runs. No commit or push has been made.


*From `GradientFlow.md` lines 1235-1246, moved 2026-10-30.*

## Bipolar meanings and the committed-row predictor (October 7; superseded by the combined step)

The order-zero
bootstrap and bilattice rules above supersede the empty/signed-complement
statements in the historical October 6 sections. Sentence identity is fixed,
recomputable and detached. The expectation predictor now takes one update
on the committed trial's rows only, using the reconstruction keep mask;
comparison costs still inspect both trials at the same parameter version.
The two answer-reader rules are unchanged. A narrowing `not` can now
change a root's meaning and earn credit through the answer term.


*From `GradientFlow.md` lines 1247-1285, moved 2026-10-30.*

## The occurrence's address (October 7; isolated historical measurement)

Sentence identity is a source address, not a presentation counter. The writer
hashes the document key, sentence index and content key into a full 64-bit
address, stored as an int64 bit pattern. Content identity hashes the identified
word bytes in order; it excludes document, model clock, composed vectors and
random state. `refs`, predictor pairs, object references and thought provenance
retain addresses while the derived address-to-row index follows compaction.
Definitions use `(DEF, word, object)`; produced thoughts use their source turn
and ordinal. The retired occurrence counter and random store namespace have
no role in new writes.

The `.when` band is the source-relative unit bracket `[i, i+1]`, encoded by
an endpoint sum on the two spatial-scale quadrature rungs. The exact model
clock goes to `timestamp` only; recency still reads that column. The complete
reader inventory
and its test fixture distinguish value readers from carriers and Teacher's
separate objective-address API.

*Reversed: this per-document `.when`, decided 2026-10-03 against a global clock that moved trained answers, returned to model time for stored entries (2026-10-20, 2026-10-23) and then everywhere (2026-11-10), with both clocks put back at each epoch boundary ([Architecture](Architecture.md#one-where-one-when-many-whats-the-field-attention-and-the-two-modes)).*

A repeated address replaces the detached committed vectors, refreshes the
clock timestamp, and combines each evidence pole by maximum. It cannot add a
second occurrence or keep obsolete leaf postings. Only a new address appends;
capacity exhaustion raises and requires forgetting. No optimizer, reader,
trial choice or gradient boundary changes in this step. Meanings and E remain
empty/zero on the fixed-form gate configuration. The separate meaning implementation
is preserved pending this review.

The isolated measurement retains four sentence rows throughout all 400
sum/XOR epochs (8/1,024 rows including DEFs). Reconstruction and raw MM pass
10/10; class passes 9/10. The additive sum control passes 10/10, with 9/10
within the final ¼ band. The two MSE misses show late common-offset reader
errors; the measurement does not establish a causal `.when`-reader attribution.
All forty final XOR roots are conjunction, R/E comparison differences remain
zero, and narrowing carries no advantage while meanings remain empty. The
raw MM path retains four DEFs and has no completed sentence closing. These
deviations are held for Claude's review; there is no retry, replacement or
commit.


*From `GradientFlow.md` lines 1286-1339, moved 2026-10-30.*

## Combined step: source address, meanings, centroid and membership priming (October 7; uncommitted review candidate)

The combined step supersedes the
separate meaning and address handoffs. The source address is mandatory. The model.xml
switches `meaningWidth` (zero disables bootstrap), `symbolCentroid`, and
`membershipPriming` default on; disabling bootstrap preserves the reserved
vector layout so attribution does not silently change the model's parameters.

Membership priming joins the existing concept edges in one degree-normalized
outflow budget. Word energy reaches containing sentence nodes and returns to
their words, detached and forward-only, through the existing retrieval prior.
It adds no per-word relevance weight. Attention remains the `.where` mask
walk; its omitted target cost remains in the owner-step comparison.

The expectation predictor learns only from the committed trial's row. Trial
comparison still reads both costs before updating. Form normalization and
meaning composition are separate, including for parameterized structural
maps: their form input has a zero complement, and an undeclared meaning write
passes the designated operand's meaning through.

The combined measurement has class 10/10, reconstruction 9/10, additive sum
10/10 with 9/10 at ¼, and MM 10/10. Both grammar semantic certificates are
exact; one passing XOR run finishes with mixed connectives. Every sum/XOR
training stores four sentence rows, each witnessed 400 times, plus four DEFs.
The full sweep initially had 20 failures; the retained focused repairs pass
250 cases with 18 skips before the thirty trainings. There is no replacement
sweep or standing training.

R is not identically zero: XOR has 1,221 nonzero reconstruction-cost
differences on narrowing departures. The last twenty epochs' 47 `not`
departures all receive credit against the departure, with contributions from
both R and the answer. E stays zero. On the failed XOR-09 entry, switching
meanings or the centroid off restores reconstruction; switching priming off
does not. All three individual controls restore sum-08 to the ¼ band. The
paired MM comparison also changes beyond RNG: the relative band stamped by
`PartSpace._embed_radix` changes its first forward. These are review findings.
No commit is made; the independent-components work awaits review.

The first repair sweep completed 5,354 cases with three failures.
The authorized repair now has a green 5,500-case sweep
on its exact measured source, followed by 10/10 sum at ¼, 10/10 XOR class
at zero, 10/10 reconstruction and 10/10 MM_xor. R and E are identically zero
on all 64,000 grammar training trial rows. Final XOR operators are 36
conjunction and 4 disjunction. All narrowing credit is from the answer;
`not` receives credit against 753 departures, for 289, and ties on 57.
Priming remains in [1,2]; every grammar store ends training with four
sentence rows and four DEF rows out of 1,024. Alec
accepted the repair as the landing on October 7.
Independent components follow, with magnitude as definedness first.


# Training.md

*From `Training.md` lines 92-112, moved 2026-10-30.*

The enabled sentence predictor defaults to distinct local NP1/VP/NP2 targets
and an occupancy objective. Empty roles have zero targets and remain in
the MSE. Closing gain and object masks affect conceived evidence only, never the
predictor objective; [ExpectationRetention](ExpectationRetention.md) specifies
the subtraction, row surprise and residual query credit. The root-only predictor remains an explicit
benchmark option. Prediction targets and durable history are detached;
preceding source encodings within a training step remain live under the
objective-local state boundaries in [GradientFlow](GradientFlow.md). Cursor
document addresses reset the affected transient
prediction view without discarding already-scored pairs. No supplied-answer
label is manufactured by this objective. Expectation is on by default, with
`interLossWeight=0.1`, `armaScale=0` and `interContrastiveWeight=0`.
BasicModel now selects tied completed-input reconstruction.
Full nested meaning and reasoning remain separate acceptance gates.
See [Layers.py](../bin/Layers.py) and
[Models.py](../bin/Models.py).

*From `Training.md` lines 748-759, moved 2026-10-30.*

Inter-sentence MSE/contrastive losses are consumed independently of Teacher's
legacy ARMA/intra gate. Prediction sees a bounded per-row view of external
predecessors, with detached preceding context and detached observed targets;
compiled and eager packed boundaries score each observed pair once. Durable
LTM remains detached. Expectation is now on by default and predicts local roles. Retained nested
meaning and the reasoning migrations remain pending.
The observation and cleanup code is in
[Layers.py](../bin/Layers.py) and
[Layers.py](../bin/Layers.py); the training gates are in
[Models.py](../bin/Models.py) and
[Models.py](../bin/Models.py).

*From `Training.md` lines 761-769, moved 2026-10-30.*

SBOW loss uses the same negative-sampling objective, with $s(a, b)$ the
wrapped-MSE torus similarity (`_wrapped_mse_score`) rather than a dot
product:

$$\mathcal{L}_{\text{SBOW}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \sigma\big(s(c_i, v_{w_i})\big) + \frac{1}{K}\sum_{k=1}^{K} \log \sigma\big(-s(c_i, v_{w_k^-})\big) \right]$$

This SBOW formulation is historical. Distributional code
pressure is now off: reconstruction alone owns the codes. It uses
momentum SGD; expectation predictors keep their separate Adam state.

*From `Training.md` lines 974-1286, moved 2026-10-30.*

## Historical reconstruction objectives and migration (2026-09-12)

This section records the September 12 implementation and measurements. Its
student default, inverse fallbacks, output stamps and placement claims are
historical. The student runtime, D3 path and old mode selectors were
retired. The one-way loader migration remains because `data/BasicModel.ckpt`
contains twelve student parameter keys; no runtime recreates that student. The current objective and gradient contract are described under
[Within-sentence reconstruction objective](Training.md#within-sentence-reconstruction-objective) above.
The tied migration's native measurements and full-suite verification are
complete.

What trained on the serial per-word path at that date:

- With `<detachedReverse>true</detachedReverse>` (the production
  `BasicModel.xml`, with `<teacherReconstruction>`), `lossIn` is the
  detached idea-only student `ReverseConstructionChooser`
  (`_detached_reverse_construction_loss`): its own parameters (an idea
  projection, a per-step slot embedding, a kind head and a rule head)
  predict the ReconstructionStack's *detached* arity, rule and leaf targets
  from a *detached* root idea. No gradient reaches the forward through it.
  `reverseReconstruct` is deduplicated out of the training step and serves
  evaluation (the trace-driven un-fold and the stage-walk reverse).
- Without it, `lossIn` is the D3 per-word reconstruction objective
  (`_d3_reconstruction_loss`, `reverse(S)` from the root scored against
  the unmasked input), or the masked-event loss where neither applies.

Declared migration:
`<reconstructInLoop>` selects one bounded compiled traversal of the
completed sentence's retained derivation with the compose path's tied
inverse transforms (the existing `invertible=True` forward/reverse
pairing). The first part (2026-09-12) is in: after the final closing, a
`torch.while_loop` replays the recorded choices backward from the root
(`_reconstruct_sentence_traversal`; the reverse steps
`LanguageSpace.reverse_binary_step` / `reverse_unary_step` through the
gate-free `generate_functional` inverses and the exact residual for
`chunk`/`sum` against the word's dictionary row), recovers each word's idea
at its position, and scores it against the word's retained reference (the
leaf the forward pushed: the word's symbol row's dictionary atom, its
object concept's where it has one, scaled by the word's signed activation;
built after the loop from the staged atoms and the loop's activation slab,
not carried, since the loop's autograd stacks every carry once per step;
until 2026-09-14 the reference was the unscaled word atom while the
forward folded the object atom, so the residual reverses were measured
against the wrong leaf); that cost is the idea-level diagnostic, and a per-row truncation flag reports a
derivation that did not account for a word. The second part (2026-09-12) adds the
byte-level fidelity that is `lossIn`: each recovered idea is softly assigned
(`softmax(|cos| / 0.1)`: a symbol's value is its signed activation and its
identity its row, so the snap ignores the sign) to the sentence's own word rows (the retained
constituent references, bounded by the word width), the assignment's
expected bytes are scored by cross-entropy against the word's own bytes over
its byte window (the percept store's byte atoms), and the gradient flows
through the assignment into the recovered ideas and the tied inverses. A concept
row is an identity code, not a fold of the word's byte atoms (concept
rows are relational identities; the byte fold lives in the perceptual
ladder), so the tied inverse of the concept lookup is the snap to a row
and the inventory row belongs to the object. Candidate surfaces come from
the surviving DEF rows: the derived index resolves `object → words`, and
the word's definition metadata supplies its UTF-8 form. Neither operand's
learned vector is compared to find that link. `word_surface_for_row()`
returns no surface for an unresolved lexical ambiguity
([Spaces.py](../bin/Spaces.py)). The brick's bounded dictionary snapshot contains
only its staged object rows (`_ar_concept_lookup_rows`) and resolves
their candidate bytes through that index (`_ar_bank_bytes`); input part IDs
provide the byte-window shape and scoring targets, never candidate bytes
([Models.py](../bin/Models.py)).

Definition metadata persists alongside the common store's tensor state.
Load and compaction rebuild the same derived index; old META bindings
migrate into DEF rows. The independent row-spelling caches and their
`concept_word_surfaces` writer are retired, so forgetting a definition
cannot leave its spelling available through a second cache. Strict load
also materialises a saved lazy chunk prior before the key audit, so no
preparatory input pass is required ([Models.py](../bin/Models.py)).

A missing WORD surface removes that candidate. A missing snapshot never
falls back to the presented words' bytes. A null candidate of similarity
zero and uniform bytes remains available alongside admitted candidates.
A sentence with no admitted candidate contributes no term, is excluded from
the reconstruction average and is counted as unavailable. Its zero cost is
not a fidelity success ([Models.py](../bin/Models.py)). Rows absent from the snapshot do
not enter the score. The
score is taken at the pop step, inside the traversal, and the loop carries
only the running sums (idea cost, byte cost, word count) besides the
reverse stack: the loop's autograd stacks every carried tensor once per
step, so a `[B, W, D]` slab of recovered ideas in the carry did not fit the
accelerator at the production width; the slab is kept only when
`_recon_keep_ideas` is set (the tests' diagnostic), and the expected byte
distribution is accumulated by scatter, never as one-hot tensors.
The reconstruction runs in two bounded passes that share the forward's
word index (`_reconstruct_sentences`): pass A, one `torch.while_loop` trip
per sentence slot up to the row's highest sentence id (a tensor bound; a
host sentence count specialised the graph once per distinct count, a
recompile of about 45 s on every brick), un-endings each sentence from its
end state: the top three STM slots and the depth at the sentence's end
(a relative sentence keeps its depth-3 end state, the three LTM slots; an
absolute one its single root), carried per sentence by the word loop for
intermediate ends and read from the ended buffer for the row's last
sentence (the live root the loop stored at
the sentence's intermediate end; `S` after the final closing for the row's
last sentence) through the recorded closing binaries into its pre-closing stack;
pass B is one `torch.while_loop` over the word index, latest word first,
that loads a sentence's pre-closing stack at the word that ends it, undoes
the word's recorded unary, post-binary and pre-binary folds, and pops and
scores the word. Operand identity: the trace records each
binary fold's operand concept rows (left = STM slot 1, right = slot 0,
read before the reduce moves the stack, including every intermediate
packed sentence closing; `record_choice` on the eager path, two bank slabs
on the compiled one), and an undo routes the
residual reverse to the operand that is a word of the sentence, on
whichever side the fold put it, with that word's retained reference; a
compound operand (a composite folded earlier) takes the residual. A fold
recorded without rows keeps the positional fallback: the endings fold
newest-first, so the k-th closing undone, last first, returns word `lo + k`
on the left, and a per-word fold's known operand is the pushed word on
the right. Where the recorded op is declared lossy (the set ops,
`part`, `whole`) or a balanced split (`lift`, `lower`), the recovered
ideas are not the words: the fidelity of those derivations is what the
byte cost measures and trains, not an exactness the reverses could
promise. Under the tied contract no legacy reconstruction runs: the
per-word reverse-from-S objective and the training call to
`reverseReconstruct` are skipped. Costs accumulate per sentence slot (`[B, slots]`,
reported) and the row cost is the mean over the sentences present. The
earlier form, one traversal per sentence over a schedule the width of the
row, cost a sentence count times the loop steps of the forward (about
150 s per brick at the production width); the two-pass form costs one
loop step per word. Measured on `data/BasicModel.xml` with
`<reconstructInLoop>` (B = 4, 400 documents, MPS, the eager compile
backend), the reconstruction adds about 0.7 s to the 8.4 s forward and
2 s to the 20 s backward of a batch. `BASICMODEL_RECON_PLACEMENT`
(`graph`, the default; `compiled`, its own compiled call after the
forward; `eager`) selects where the traversal runs, for the performance
protocol.

The answer-materialisation boundary (revised 2026-09-28).
`Understanding.sentence_states` owns each row's final `SentenceEndState`;
`sentence_fields` indexes every packed sentence. Each record holds its actual
one-slot or three-slot meaning, references and field coordinates, cloning
values while retaining current gradients. A selected question retains its
typed semantic request. Completed records contain no leaves, actions or
compose program ([Understanding.py](../bin/Understanding.py)).

The temporary `AnswerProgram` exists only while reading. Its numerical journal
supplies actual pre-fusion operands for prediction and cached reference fields.
Reconstruction inverts the chosen operations there, and exploration uses the
record to force a different choice. The sentence driver discards these records
after scoring and committing the chosen end state. Raw forward and evaluation
use this same driver with one exploit path and no optimizer.

Resolution selects a current field, a detached recalled field, a prediction,
or a checked thought result. It generates from that structure. Lexical
reference questions use the native concept index; they do not search a saved
input leaf list. `AnswerDerivation` owns the resolved conceptual answer and the
question's target-free context ([Output.py](../bin/Output.py)). Discourse
retains detached sentence fields including packed slots, so later staging
cannot replace a held answer ([Models.py](../bin/Models.py)).

`_materialize_answer_idea` applies one conceptual-width conditioner to the
resolved root, once. Missing fields are explicitly unresolved; a future
prediction without a conceptual predictor has no completed field
([Models.py](../bin/Models.py)). Replaying the identified compose actions recovers
the idea; output chooses its own generate derivation. Reconstruction
targets remain metadata and never choose output actions
([Models.py](../bin/Models.py)).

Both output modes consume the full-width concepts directly. With
`outputInLoop`, the three opaque slots enter a bounded traversal, newest
at slot 0; the generate walk emits concepts for the shared reverse chain
([Models.py](../bin/Models.py), [Models.py](../bin/Models.py)). Otherwise, the conditioned
concepts enter the dedicated conceptual synthesis operator, then the shared
reverse body and perceptual inverse, the dedicated perceptual operator,
and the output adapter ([Spaces.py](../bin/Spaces.py), [Models.py](../bin/Models.py)). The
native 1032-wide concept / 136-wide percept path needs no dense symbolic
seed; topologies without row programs retain the dense compatibility path
([Models.py](../bin/Models.py)).

Historical conditioner widths remain in the checkpoint registry. Strict
loading restores their exact weights and initializes a missing active
conceptual width to zero when migrating a narrow-only checkpoint
([Models.py](../bin/Models.py)). Already materialized answer parameters join a newly
created optimizer; later lazy modules join the live optimizer once
([Models.py](../bin/Models.py), [Models.py](../bin/Models.py), [Models.py](../bin/Models.py)).

Realized-answer supervision (2026-09-15).
Answer preparation is explicit: `resolveAnswer(understanding, questions)`
returns the owned, target-free derivation accepted by `reverseOutput`.
Generation cannot rerun answer/query resolution, and the conceptual handoff
retains gradients within the current optimizer step
([preparation](../bin/Models.py),
[realization](../bin/Models.py), [owned value](../bin/Output.py)).
Training accepts only available, separately supplied answers to supervised
questions. It filters rows before selecting numeric or text scoring, so an
automatic text target cannot displace a supplied numeric label. For text,
the fixed answer percepts are realized in input-event space without desired
content, then scored against the embedded supplied text
([Models.py](../bin/Models.py)). Evaluation retains automatic present/past/future
metrics. Missing labels and automatic input targets provide no dedicated
answer gradient or optimizer update, including after an earlier Adam step;
thinking policy credit also excludes these rows, including closure costs
([Models.py](../bin/Models.py), `test/test_output_path_supervised.py:284`). Shared
understanding parameters can still learn input reconstruction.

The output walk's generate policy (2026-09-15).
`LanguageSpace.generate_policy` is a linear chooser created only under
`<outputInLoop>`. Its parameters belong to SymbolSpace's explicit optimizer
list. BasicModel enables it with `outputPolicyWeight=1.0`; unlabeled FineWeb
still gives it no answer credit. A training `reverseOutput` samples its own
actions from the chooser;
it does not follow the input's identified compose derivation. The loop
returns the sequence negative log probability of those choices. It draws
one fixed-shape random slab before entering the loop and detaches the
policy's top-slot features, so policy credit trains the chooser while the
ordinary realised-answer error trains the selected numerical transforms.

`runBatch` resolves supplied desired answers after generation, then applies
answer-error credit only to available `What.supervised` rows. For row `b`,
let `E_b` be the realised output's error against that supplied answer and
`C_b` the sequence negative log probability. The credit is
`mean((-E_b - baseline) * C_b)` over eligible rows, with detached errors and
the previous exponential moving return baseline (initially zero; updated
with decay 0.9). `outputPolicyWeight` multiplies this term in the actual
training total, and `output_policy` reports it. Zero weight, missing
supervision, present/input targets and prediction-only trials supply no
policy gradient. The previous batch's cost is cleared at entry even when
the next batch skips output generation. This follows the answer-error
credit pattern used by the thinking chooser; no gold parse is assumed.

Evaluation uses the chooser's highest-scoring action. The rule inventory
comes from the grammar's `<generate>` section, with the number of LHS
outputs determining binary/unary expansion; it can differ from `<compose>`
and can contain rules absent there ([Language.py](../bin/Language.py)). The numerical
inverse kernels remain shared with reconstruction, while the catalogs,
policies and traversal state are independent ([Models.py](../bin/Models.py)). An
explicit output-owned generate stamp can replay an output rule; an input
compose stamp cannot. The low-level `targets` argument is retained only as
an ignored compatibility argument. There is no `outputTeacherForcing` knob.
Reconstruction continues to follow the input's identified compose parse;
that parse is not assumed correct and provides no output imitation credit.

Checkpoints record stable generate-action keys. Loading maps chooser rows
and Adam moments by those keys, preserving stop and giving new actions
fresh weights and zero moments. Older checkpoints without keys map from
the legacy compose catalog ([Language.py](../bin/Language.py),
[checkpoint_migrations.py](../bin/checkpoint_migrations.py)).
A completed constituent is popped into the emitted sequence and the walk
continues with its pending constituents. A row completes when no live slot
remains; exhausting the budget with pending slots reports truncation.
Emitted words are returned left to right.

The eager trace replay (`_reverse_reduce_unfold`) and the exact leaves
teacher (`_reverse_method1_leaves`) are retired (2026-09-13): in
evaluation `reverseReconstruct` with `<reconstructFromIdea>` takes the
sentence's end state and unwinds the recorded derivation through the
same traversal (`_recovered_word_ideas`), then realises the recovered
per-word ideas through the reverse chain.

The grammar ops treat the concept event as opaque (Language, "Concept
events are opaque to the grammar ops"): the lift/lower inner layers are
sized to the muxed concept width, so checkpoints written before
2026-09-13 re-initialise those layers on load.

Loop gradients (2026-09-13). The tying gate found that the reconstruction
cost reached the references but not the fold weights, and the cause is in
`torch.while_loop`'s autograd (torch 2.14 and 2.15 nightlies): the
per-trip checkpoints inherit the initial carry's `requires_grad`, so a
carry that enters a loop as plain zeros returns a zero gradient from every
trip. The chain across trips is cut, parameters used in the body are
credited from the last trip only, and closures loaded mid-loop get
nothing. The forward word loop was affected too: on the ladder fixture 26
of 42 parameters received no gradient through the loop and 13 more a
different one. Every loop's carries now pass through
`Models._carries_with_grad`, which adds a zero scalar leaf that requires
grad (a per-device anchor created eagerly, since a tensor factory with
`requires_grad=True` cannot be traced inside a compiled region); with it
the loop's gradients equal a plain Python loop's on every parameter.
`test/test_while_loop_gradients.py` pins the defect and the fix. The
same node also keeps its per-trip checkpoints alive after the brick
(it outlives the brick through the C++ graph), so
`Models._release_loop_checkpoints` drops them at brick entry, after the
optimizer step, walking only the previous brick's own graphs (from its
published tensors), so a pending graph held elsewhere keeps its loops; the STM's live state and the trace's loss slab, both
updated in place, are detached there too. The traversal is part of the tensor word
pipeline's sentence state (the compiled path and its eager `while_loop`
form); the legacy static scheduler produces no such state, and `lossIn`
there falls back to the D3 objective. A checkpoint carrying the detached
student's parameters loads under the tied contract with those keys dropped
(reported), and a tied checkpoint under `<detachedReverse>` rebuilds the
student fresh. The third part (2026-09-12): with `<outputInLoop>`, `reverseOutput`
runs the resolved answer's generate walk as a second, separately compiled
`torch.while_loop` (`_output_generate_walk`): the top slot's `.where`
stamp is decoded in tensor form, a rule stamp applies that rule's tied
inverse (binary opens the slot above; unary in place), children are
stamped empty as the eager `unreduce` does, and the walk ends when no row's
top is a rule or the budget (`N - 1`) is spent (reported truncation). The
walk owns its state and termination and reads nothing of the input
reconstruction; the spaces' reverse chain and the output decode follow it
unchanged. It
is a change of learning contract, so before `<detachedReverse>` retires:

- objective: sentence reconstruction fidelity (byte cross-entropy over each
  word's window) plus the diagnostic idea-level cost, reported separately
  from linear-inverse accuracy;
- parameter ownership: the traversal owns no parameters; it moves the tied
  transforms and the forward's fold parameters; the student's parameters
  leave the state dict;
- gradient boundaries: the cost stops at the recorded discrete choices
  (credited through the existing policy credit) and at constituent
  references (indices); the two knobs are mutually exclusive;
- checkpoint migration: loading a checkpoint that carries the student's
  parameters under the tied contract ignores them; the reverse direction
  (tied checkpoint under the detached contract) rebuilds the student fresh.


# ThoughtHistory.md

*From `ThoughtHistory.md` lines 61-89, moved 2026-10-30.*

## Evidence

The reviewer probes first failed against the pre-installation state. The
focused contract set then passed 9/9, its expanded lifecycle/checkpoint set
passed 33/33, and the affected integration selection passed 69/69. These
prove the storage, replay,
credit and occurrence-read foundation only; they do not prove normal thought
selection, learned utility, or end-to-end answer quality.

The later typed-result retention probe first failed because a `ThoughtRecord`
had no result field, and its prediction-shaped boundary probe then caught
live `MeaningExpectation` tensors. The detached restore, prediction,
integrated checkpoint, and v1-compatibility cases each passed 1/1; the final
bounded controller/history/query selection passed 134/134. This is a
typed-boundary and replay result only, not
evidence of learned controller utility.

The nested-result reviewer probe then failed because v2 serialized a complete
meaning as an ordinary mapping. The v3 tag restores
the typed detached meaning while accepting v1/v2 sidecars; the focused repair
passed 1/1 and the full history-boundary file passed 14/14. The affected
controller/history/query selection then passed 136/136.


# AccessibleMind.md

*From `AccessibleMind.md` lines 161-182, moved 2026-10-30.*

The deterministic generativity probe uses distinct concept codes in each chain.
It writes one compound per condition, keeps
its stored idea fixed, then unfolds it with the current generate MLP and shared
operator after 0, 1 and 8 small training updates. Original trees and leaf codes
are used only to score the result; they are not unfold inputs. It compares both
codes and operations.

| Chain length | Composition depth | Exact codes and derivation at updates 0 / 1 / 8 |
|---|---|---|
| 1 | 0 | 1/1, 1/1, 1/1 |
| 2 | 1 | 0/1, 0/1, 0/1 |
| 3 | 2 | 0/1, 0/1, 0/1 |
| 5 | 4 | 0/1, 0/1, 0/1 |

These null results establish no learned compound generativity or chained-episode
recovery. Those historical measurements used the earlier recorded-leaf index.
Under the September 28 decision the index itself depends on generation from
the actual field, making weak generativity a retrieval limitation too. The
historical values are retained; they are not a passing learning claim for the
new index. The small probe
is also not a corpus estimate, multi-seed study or causal-utility comparison.
[Probe](../test/test_mind_generativity.py).

## Retired figures (2026-11-01)

These figures describe designs that are no longer built. They are kept as
the record and are no longer generated; the current figures are produced by
`doc/diagrams/gen_diagrams.py` and the `fig_*.py` scripts beside it.

![MM_5M, the hierarchical progressive bottleneck (retired)](diagrams/mm5m_architecture.svg)

MM_5M held $N \times D = 4096$ constant across eight conceptual levels. It
was the reference figure of [Architecture](Architecture.md) and
[BasicModel](BasicModel.md) until 2026-11-01. No configuration defines it
now: the 2026-10-02 configuration review deleted its last variants. The
reference figure is now generated from `data/BasicModel.xml`.

![MM_5M_bivector architecture (retired)](diagrams/mm5m_bivector_architecture.svg)

The bivector variant of MM_5M, drawn in May 2026. Its generator was never
committed, and no configuration defines the model.

## Work recorded in todo.md until 2026-11-03

*Moved here from `todo.md` on 2026-11-03, when each of these blocks collapsed to a Done line after an audit of its remaining work (the audit's outcomes are the Done lines and the "Carried ... 2026-11-03" paragraphs in `todo.md`). The blocks are as they stood, in their original order; their links are written relative to the repository root, as in `todo.md`.*

### Mature-checkpoint learning evidence, as it stood in todo.md

- **Mature-checkpoint learning evidence.** The mechanism, checkpoint exposure
   counter and quality-evaluation machinery are reviewed and implemented. Learning
   acceptance remains deferred until a checkpoint has at least **one million
   completed FineWeb training sentences**; missing exposure skips, never passes,
   the quality evaluation. Correctness checks stay unconditional. No qualifying
   checkpoint has been evaluated. Then run the loaded-model wording, prediction
   and predictive-thought checks; the full causal gate still needs separately
   trained, equal-update controls across at least three seeds, with tasks, run
   length and thresholds declared first. Ordered prediction must beat shuffled
   and context-free controls without worsening reconstruction/discrimination
   against reconstruction-only; thought must improve error at matched work or
   reduce work at matched error. Same-checkpoint ablations do not establish this
   causal result, and fused AMP conservatively undercounts exposure. Preserve the
   tiny-run nulls, incomplete reverse programs and 5/56 wording failures as
   development diagnostics;
   old context-free and assertion-thought scores are invalid controls.
   **The independent-components learning gates** — held-out anaphora against `206a0146`, verb
   reuse, prediction control, shuffled order, renamed vocabulary and
   determiner control — remain pending the million-sentence checkpoint,
   across **seeds 0/1/2**, and are **not claimed** by its mechanism acceptance.
   **The math-chain learning measurement** (MM_math_chain: held-out binding 9 of 10,
   the chain bar, the expectation-only and zero-budget controls) also waits
   here, at the full training session's checkpoint:
   one-step problems first, worked steps as intermediate answers, binding a
   found candidate, and budget in steps with each query's record charge bounded
   separately.

### Learned utility and structural preference, as it stood in todo.md

- **Evidence: learned utility and structural preference.** Prefer
   understandable structural operators when they carry the meaning; any opaque
   operator must be an ordinary grammar-MLP choice, with the structural face
   preferred at equal fit. Measure the opaque routing share and its decrease
   as structural coverage grows on the same corpus. Held-out causal utility
   against direct-answer and equal-compute baselines across seeds;
   reconstruction/discrimination controls; warmed training throughput; the
   preserved arbitrary-symbol poison probes and renamed-vocabulary controls.
   Numerical values or symbol IDs never supply learner arithmetic or answer
   seeds. Learned utility stays explicitly unproven until these comparisons
   pass. The seed audit of `d4dc385` left the unseeded MM-grammar XOR gate
   failing (.21757 after 900 epochs, bar <.20), and the two `XOR_grammar.xml`
   CLI gates stopped before training at unsupported W=6. Keep these failures
   visible; no passing-seed selection or expected-failure waiver.
   *Compatibility:* `XOR_grammar.xml` composes `not` / `conjunction` /
   `disjunction` over symbols on the serial path and never used the deleted
   tower folds, so its gate stands as written. Reconstruction comparisons use
   the reviewed shared-fields baseline:
   serial before/during/after .1005906649 / .0948241442 / .0928765051;
   packed/single byte reconstruction .6839025617 with exact parity. Learning
   quality gates follow the million-sentence prerequisite; mechanism
   and regression checks remain unconditional.
   The implementation adds
   exact-score tie preference and committed-program routing measurements;
   its protocol predeclares the
   qualified causal study. Learned utility and declining opaque routing remain
   unproven. W=6 scheduling now reaches the next unchanged
   CLI blocker: the six-row symbol inventory exhausts during the first epoch's
   reset/autobind, after batch execution. Keep both CLI assertions and the
   historical MM failing-seed result visible. All three declared diagnostic
   seeds were attempted; seed 1 hit the 8 GiB guard and remains incomplete.
   The initial 4,968-case full sweep exits 1 on an unchanged strict-XPASS
   reconstruction marker. Claude's review corrections
   use the marginal uniformly and require complete reconstructed positions;
   the marker and threshold remain intact. The corrected source passes its
   single 4,980-case full sweep: 4,653 passed, 326 skipped and one expected
   failure. Reconstruction exactly matches the reviewed shared-fields baseline. The review corrections
   are validated for the requested commit; the empirical gates remain open.
   The MM_xor bar stays as written: its bests sit within 10% of the bar in
   every campaign, and the rule — a miss whose path differs
   from the landing only in global-RNG consumption, shown by bisection, is
   not a regression of the candidate — is what handles that margin.
The non-blocking compose follow-ups of one operation per round (the training
temperature and parsimony term) are in
[FutureWork](FutureWork.md#compose-follow-ups-2026-09-27);
its 2026-09-29 costing finding is closed by scoring both sentence trials before
either trains (XOR_grammar, September 30), and the depth-three
campaign stays a recorded red in the
accepted landing, not work.
Remaining operators work stays where it was assigned: surface, tense,
morphology, aspect, null and situation codes with occurrence, tense and surface
form; forgetting's truth-kind bias with forgetting; trained-checkpoint
evaluation with the run harness; the concept-face negative
image's learning measurement with the mature-checkpoint evidence (its mechanism landed at
`3de37eef5`, and the complement it reads is populated since the October 7 landing); the
open-read/teacher-loss host island with the compiler work; expectation across the
architecture as separate work; the bank's collision and merge rules in
[FutureWork](FutureWork.md#one-bank-the-collision-rule-for-property-rows-and-a-merge-rule-decided-2026-10-06-unscheduled). The
[FutureWork list](FutureWork.md#left-unbuilt-by-the-one-attention-2026-10-04)
retains the dynamic stop, the remaining operator names and other deferred work. The
reading residue around forced `interpret` and digit boundaries remains open.
Deferred from independent components (non-blocking): the detached role view's
retirement is with housekeeping; learned-column pruning is with forgetting.

Alongside occurrence, tense and surface form, retain the closing mechanism's open learning findings:

- Stored-idea recovery beyond two words remains a learning gate after
  the constituent lesson; no noun-side sparse-coding mechanism remains. Initial-source list free recovery is 0/4 throughout; enabled
  two/four-word derivation recovery is 2/4 and 0/4. The order lesson's CE
  stays at .3465735912 / .7945134640 through 64 presentations, with zero
  scorer gradient. These inverse and credit findings remain active work.
  The valid part-2 two-word checkpoint reads its stored derivations at 4/4,
  but free readback is 0/4. A read-only probe
  finds every attention tanh unit saturated at ±1 and candidate logits tied;
  that study left saturation unchanged. The bounded-context candidate
  bounds the shared context and removes saturation in the retained 64-step
  continuation. Fresh two-/four-word CE improves to .0320/.0166 but remains
  above .01; free recovery remains 0/4. New adverse result: the later relation
  trace falls from 4/4 to 2/4, swapping `than` and the final operand.
  Pair teaching also skips compound operands without identities in its word
  shortlist; the successful lexical-pair probe does not establish sufficient
  teaching credit for free reconstruction of longer constituents.
- MM grammar recovery: the 1,024-row LTM exhausts in epoch 2 before evaluation;
  there is no new recovery rate. Forgetting may not discard derivations yet.
- Compiled retention: the 64/312-element mismatch (max .2556) has no identified
  cause. Later passing runs do not close it. Also retain the accepted support-mask landing's packed
  LTM retention and pending-premise order-`False` findings.
- c's retained supplied answer does not match `five`; its assertion stays.
  The old eight-sentence fork diversity contract was retired by decision
  and replaced with bounded coverage on a trained artifact.
  Question-sensitive answering also remains unproven in the current staged
  warm-up: noun–verb answers are 0/4 and relations 1/4; the earlier 4/4
  identity and noun-phrase scores match a last-word baseline.
- The October 14 sweep
  has three failures new relative to the accepted forward-discard repair's passing sweep. The
  supplied-answer failure and eight-sentence fork failure (now only round
  24) recur from the support-mask landing. `test_forced_ordinary_bound_declaratives_open_no_episode`
  is new relative to that landing too: no open slots or episodes appear, but every
  recorded compose departure is 6, failing the diversity assertion.
  The October 15 source-boundary replay
  finds no exemption effect: the shared fixture has an empty meaning slice,
  so both implementations charge zero and produce identical departures and
  keep decisions under matched initialization. The two diversity assertions
  demand more than the independent reservoir/walk sampler guarantees; on the
  recorded menus their failure probabilities are 2.15% and 26.21% respectively.
  Those are conditional probabilities, not full-test failure rates. c's
  immediate failure is different: its filled reference points to an order-one
  child associated with `the`, with no lexical path to `five`. All assertions
  remain unchanged; no repair or learning claim is made.
- The forward-discard sweep
  adds `test_review17_measurement_wiring_on_one_ordinary_batch` as a new
  failure relative to both the support-mask landing and the bounded-context candidate: its observer has no epoch-1 chooser
  logit entry (`KeyError: '1'`). The accepted inverse-domain repair resolves
  the cause: the observer dropped detached training calls on narrowing-only
  departures. It now counts those separately from gradient calls; the native
  regression and unchanged ordinary-batch certificate pass.
- Per-position cleared-cache overlap remains 0.000 against .800. The
  follow-up bisection
  separates the initial self-pair obstruction (repaired in part 1) from the
  current reversed pair. The first part-2 candidate removes the exact-fit
  override: learned weights give 1.000 overlap, against 0.000 with the old
  override on those same weights; the cold probe remains 0.000. Keep the order lesson, content
  coverage, walk order stamps and family-metadata use under part 2.
- The identity corpus preparation
  exposes a native exploratory trace/admission mismatch: an admitted leaf is
  filtered from reconstruction, which then underflows a binary operation.
  The first part-2 candidate repairs the trial/commit masks and verifies the
  preserved passive-input trace; it landed as preparation in `af634ed35`. The corpus drops
  unverified object fronting and reports the introduction-role/position confound.
  This probe has not been bisected against the support-mask landing; do not call it a regression.
- The first part-2 warm-up stopped on a non-concept `sym` cue entering the
  taxonomy order reader (four-word presentation 8; all experiment branches
  unreached). The candidate restricts that reader to positive concept handles.
  The corrected run passes that point and retains a negative address tagged
  `sym`; its old-reader replay raises and the corrected boundary unfolds the
  payload. The producer's typed-reference provenance remains open.
  Native checkpoint loading also changed eight shared-codebook tensor aliases
  in the smoke probe; the experiment restores frozen tensors for matched starts,
  without claiming a production retention repair.
- Supplied-answer warm-up adds 16 lazy comparison-reader weights that native
  strict checkpoint preflight rejects before the child's load hook can restore
  them. The first part-2 run has eight failed branch starts. Its measurement adapter restores
  the child first and verifies all 874 saved tensors; production loading remains
  unchanged. This newly observed retention finding has not been bisected against
  the support-mask landing. The retained branch continuation reused the warm-up without extra exposure.
- The initial native identity experiment used a word-chunk fixture with no
  meaning complement. Its four completed zero-identification results are
  invalid admission measurements; the unfinished identity branches were stopped.
  A fresh run with 64 meaning pairs and a verified
  nonzero warm-up signal completes recovery at 0/240 on all four branches:
  no new columns or merges, and every row selects mint, so the on/off gate
  changes no eligibility. All 192 added observations have zero meaning-band
  content despite nonzero full vectors. Four frozen replays locate the loss:
  word meanings are nonzero, but selected `non` clears them and `lower`
  repeatedly keeps the newer operand. Resolve why compose keeps these lossy
  readings before claiming an ICA recovery or admission-gate lesson.
  The bounded-context replay
  still has zero closed meaning and `lower`–`non`–`non` in all 48 held-out
  rows despite a content charge of about 1.0. Four sampled paired walks
  charge both erased readings equally; this retention outcome remains unmet.
- The bounded-context matched rerun
  completes all eight branches with the same seed, budgets and revision-3
  corpus. Recovery remains 0/240; all identity branches remain 0/146 bindings
  and 0/20 row pairs. Each now resolves 150/690 mentions to written individuals,
  but no probe has a complete candidate set or a valid selected reference.
  The other 540 mentions have named missing reasons; grading has no sentinels.
  Balanced unadmitted leaves rise from two in the old retained audit to 43.
  The balanced gate contrast is no longer vacuous: off mints 58 columns versus
  on's 57, without accuracy gain. Recovery and biased training remain vacuous;
  the admission lesson fails. The retained observer audit
  accounts for the old 690 unresolved mentions separately. Solve the missing
  individuals/requests before interpreting binding and recency controls as
  learned choices. Reviewed and accepted for landing by Alec,
  with no learning claim; no rule is retired and part 2 remains open.
  The next candidate adds the forward
  discarded-meaning charge after the forced-reading diagnostic, with the
  scoped four-clause and two-branch recovery measurement (factorial/confounded,
  gate on), reusing the accepted bounded-context warm-up. Identity branches wait for
  nonzero closed meaning. Candidate `0cd221d74` on
  `codex/item6-r2-forward-discard` passes the 4/4 local keep regression, but
  both completed recovery branches have 0/48 nonzero closed meanings
  and identify 0/240 atoms. Factorial mints zero columns; confounded mints four;
  neither forms a measured kind/property merged pair. Confounded selects mint
  on 136/192 observed commits versus factorial's 192/192; the off contrast is
  unmeasured. All 96 kept readings still use `non`. No learning gain or rule
  retirement is claimed. This candidate was rejected. The
  inverse-domain repair
  restores the accepted zero-cost certificates and scopes the forward charge
  to the local field inverse's undefined coordinates. The observer's missing
  epoch is repaired by counting detached training calls separately from
  gradient calls. Accepted `348fb86c7` + `90ef40a42` landed on main, pushed
  October 14 with WikiOracle bump `066fb5d`. It retains nonzero closed meaning on
  44/48 factorial and 4/48 confounded held-out rows (previously 0/48 each),
  but identification remains 0/240 on both. No new columns or merged pairs;
  all kept programs use `verb` and omit `non`. The retained four cat clauses
  still close at zero. Both branches select mint on 37/192 observed commits;
  the off contrast and identity lessons remain unmeasured. The one final
  sweep completes all 5,790 cases: 5,502 pass, 287 skip, one XFAIL, no failures.
  No new ordinary failures versus the accepted support-mask landing or the rejected
  forward-discard candidate; cleared-cache
  overlap remains adverse to that landing's XPASS. No rule is retired.
  The scoped gate-off measurement
  reuses the same bounded-context warm-up, seed and revision-3 corpus. Factorial off
  mints two columns (60 total); confounded off mints four (62 total), versus
  zero new columns on. Both still identify 0/240 atoms and 0/24 for each
  kind, with no measured merged pairs. Programs and norms match gate-on:
  `verb` only, 44/48 and 4/48 nonzero closed meanings. The admission contrast
  is no longer vacuous, but its retirement lesson fails; the gate stays.
  The frozen singleton audit finds 16/24 zero stored meanings despite
  nonzero word vectors, and 0/58 warm-up columns aligned at |cos| ≥ .9.
  The confounded trace loses nonzero word meaning at each `verb` fold from
  a zero `the` accumulator. The separately applied search-kind exemption
  makes exact field OR pay zero forward discard while keeping kappa = 0 in
  the field reader; it does not change the preceding learning comparison.
  Candidate `c647e1506` completes the one full sweep: 5,503 pass, 287 skip,
  one XFAIL and three failures, with the new/recurring failures named above.
  Accepted zero-cost certificates, the observer and the 4/4 retained-clause
  keep regression pass. The parts-of-speech pass waits for a
  reading with NP structure; no rule is retired.
  The fresh corrected-objective warm-up
  completes the same stages 1–4 budgets with seed 20261010: **8/24** singleton
  rows carry their atoms, below the required 20; 16 remain zero despite word
  meaning norms of .2. It mints **zero A columns**, confirmed in the saved
  checkpoint. The native `black` trace loses .2 → 0 at `non` before closing.
  All 76 singleton presentations are observation-only, with one greedy trial
  per batch, so the corrected keep charge never compares an alternative on
  those writes. Both order CE gates still fail (.032051/.016668 versus .01),
  and all four final stage-4 supplied-answer readbacks are 0/4. Under the
  identity-per-order amendment, balanced order-1 identity is not run because
  the singleton gate fails; kind recovery is deferred to order 2. No
  production change, new sweep, learning claim or rule retirement in this pass.
  The positional-operands and paired-warm-up candidate
  (`6a14ba151` measured source) removes the name-to-order table and carries
  actual operand stamps through `argin1`, `argin2`, `argout1`; the order-3
  regression passes and operation-owned mint/bind/kind behavior stays.
  Its paired stages 1–2 retain **20/24** singleton atoms, so the balanced
  gate-on identity branch runs: 136 training and 166 held-out documents,
  **zero columns minted, 0/146 bindings and 0/20 row pairs resolved**.
  All 556 held-out clauses have zero closed meaning under `equal`-only
  programs; all 690 annotated references report `no_individual_request`.
  None reports an unadmitted leaf in this run; the earlier 43-leaf artifact
  is not retested. The bounded trained-artifact certificate covers **80/80**
  eligible rounds; the two batch-diversity tests leave the default suite.
  The one full sweep has ten failures, with subsequent targeted repairs
  leaving c's carried `five` structure failure; the paired zero-cost
  counterexample reproduces on accepted main. No clean final sweep or
  identity-learning claim is made, and no identity rule is retired.

The cleared-cache empty inverse was **new with the closing-mechanism candidate**, not an
unchanged failure of the support-mask landing. Part 1 repairs the candidate-search self-pair obstruction
and checks the unchanged drivability assertion. It also repairs the native
retrieval `meta`-reference error, scopes stored-read operator activations,
charges dropped support in the keep cost, and isolates captured walk inputs.

**The closing mechanism's carried learning, alongside occurrence, tense and surface form:**
**stage 4e first**, then identity matching, VP vectors taught by the individual's
before/after temporal difference, and word classes from direct operand participation
(never phrase heads). Order-2 kinds are deferred; neither a second dictionary nor
a taxonomy computation for their intension is chosen. Stage 4e's declared gate
(Alec, 2026-10-17) is ≥90% exact held-out free brackets, one occurrence for
≥95% of singular NPs with zero duplicate writes, one order-1 occurrence at
every lowered NP, and no occurrence for an unlowered description. The October 17
review replaces the 2→1 condition: the description's stamp is reported, not
required; mint/bind construct order 1 and kind constructs order 2. The lesson's
CE supplies waits and folds; annotations never replace the student's kept
program. No native identity run is authorized before that gate passes.
The stage 4e candidate completes its
144-presentation budget but fails the gate: **8/32** exact free brackets,
**0/48** NP occurrences and **0/48** 2→1 reductions; zero duplicate writes,
and **8/8** unlowered controls without an occurrence. The controls are vacuous
with no occurrence writes anywhere. Native `compound` cases are unavailable
in 160/288 teacher traces; all held-out referent stamps remain 0. The corpus,
wait/fold CE path and measurement are landed preparation, not a passing learning stage.
The constructed-order pass
repairs mint/bind/kind output stamps. Its checkpoint diagnostic finds live,
unsaturated chooser CE gradients, but no native sigma cases for compounds.
The one rerun stopped in its first commit because trial detachment dropped
the close-reference flag; that defect is repaired and reproduced on the
first batch. No held-out result or new learning claim was reached; the
full experiment was not restarted. This preparation is landed at `89621bc61`
(WikiOracle `c78c292`). The review corrects the adjective annotation: it is
`intersection`, not case-dependent noun–noun `compound`, so this corpus needs
no case lesson. The next pass trains the fold-pressure gain and uses the order
lesson's optimizer for compose lessons, with a 16-presentation CE pilot before
a conditional rerun at the unchanged budget.
The intersection/pressure pass
completes that rerun after a passing 16-presentation pilot. Every teacher
trace is available and wait/fold CE moves, but the free gate fails:
**0/32 brackets, 10/48 NP occurrences, 0/48 lowering certificates**;
22 unexpected occurrences, zero duplicates/collisions, 8/8 clean controls.
Its default sweep is green (5,466 passed, 333 skipped, one XFAIL), including
c this time; the broader carried findings remain open. **Stage 4e is not
learned; identity and the further surface-form mechanism work remain gated.**
That preparation is landed at `b09c6c3d4` (WikiOracle `c8c2cda`).
The student-state pass
adds stage 3 before 4e, targets the student's actual compose states, and
suppresses extra batch-end optimizer steps in the experiment. Final free
source order is **8/8** on the order-lesson fields, but both CE gates remain
failed at their recorded training endpoints (**.0320–.0323 / .0165–.0168**,
threshold **.01**). After repairing the native action-index mapping, the
corrected pilot passes (wait CE **5.775928→2.680222**, fold CE
**7.941597→3.586233**). The full 144/40 run still fails: **0/32** brackets,
**8/48** matching occurrences, **7/48** lowering certificates, **19** unexpected
occurrences, zero duplicates/collisions, and **3/8** clean controls, despite
**40/40** source-order readings. No identity run follows.
The requested stage-3 diagnostic uses the saved checkpoint with no updates:
two-word CE now passes (**.000224–.000227**, after the later four-word lesson),
four-word CE still fails (**.016268–.016439**), gradients are live and the scorer
is unsaturated. The saved training trace was still improving at the fixed
budget's end. No warm-up repeat or budget extension. The final default sweep
has **5,466 passed, 335 skipped, one XFAIL, one carried c failure**; exact
NOT/OR and paired zero-cost pass, without retiring their broader findings.
Work is on `main`; Claude approves the preparation landing. **No constituent
or identity learning claim.**
The balanced identity corpus and grader remain the measurement; the earlier
factorial/confounded recovery belongs to order 2. Keep the inverse, order-CE,
relation-trace, MM grammar, compiled-retention and c findings above.
The carried-invariants repair (2026-10-19) owns and repairs the paired zero-cost counterexample.
Acceptance retires mechanisms by decision,
not by a passing identity lesson.

### Occurrence, tense, aspect, the preposition and surface form, as it stood in todo.md

- **Where and when a sentence occurred; tense, aspect, the preposition
   and surface form**,
   after the closing mechanism; VP phase learning needs the deferred temporal-difference teacher;
   one check-in (Alec,
   2026-09-30). **Relative `.when` and address upsert moved to operators
   landing (accepted October 7)**:
   source-relative `[i, i+1]`, stable document/content address keys,
   re-witnessing and fail-loud capacity.
   The 10-03 amendment keeps `.when` in percepts and concepts; the absolute
   model clock is written to the `timestamp` column by the occurrence address (October 7) — this item
   decides its fate with the recency use
   (Alec's 09-30 age ruling: the document index suffices). Remaining here:
   documents as situation codes in content; learned before/after; tense,
   aspect, prepositions and surface transformations. Interleaving is
   delivered by one attention's open read (`serial`/`modeSchedule` are
   rejected at load); tense and aspect are
   prepositions of the verb phrase with none written, the
   compound selecting among the verb phrase's phases; the preposition is
   one operation with two attachments; markers are leaves, minted on
   recurrence, and `surface` is one operator with four suboperations,
   trained by reconstruction under parsimony. Removes the old tense, aspect
   and morphology code and its table of English forms. Dimensional
   `lift`/`lower` is not here (FutureWork; Architecture's statement that it
   remains here is corrected under housekeeping). Exit: the remaining mechanism tests;
   learning measurements after independent components, recorded and not tuned.
   Replace the legacy `get_stm_chain` recency across batch rows and test
   lookup by the row's occurrence address. The later Orders theory
   governs order operations.
   **Sequence (Alec, 2026-10-17):** stage 4e alone first, with its fixed budget.
   Stop at a failed gate.
   The separate mechanism work retires the legacy clock timestamp and
   `Clause.eternal`, replaces recency with the row's source-addressed chain,
   and replaces the analyzer with learned surface transformations (two faces remain in the grammar: absorb and transpose)
   and their inverses. Measure the clock-independence,
   offset-independence, addressed-chain, interleaving, preposition and
   surface-inverse certificates. Tense/aspect on VP phases waits for the
   carried identity sequence. **Remainder collected as the mechanism residue (Alec,
   2026-10-19), after the run harness.**

### Run harness and resume test, as it stood in todo.md

- **Run harness and resume test.** *Once pulled forward for the
   conference (Alec, 2026-09-27; no freeze since 2026-10-03, so it is built
   with one attention and evaluated here):* the word-level predictor as the NanoChat gate's
   evaluator — the model's own expectation of the next word scored on the
   frozen item manifest (top-1, reciprocal rank, shuffled-prefix control),
   living in `eval_nanochat_grammar.py` rather than the training loop; it
   is the first level of one attention's expectation at every bracket. Also logged
   per bracket level once one attention lands: the both-rate and the
   categorical-discrimination index. One logger per interval: reconstruction
   loss; expectation discrepancy; LTM occupancy and forgetting passes;
   [categorical discrimination](../bin/CategoricalDiscrimination.py) from the
   fixed four XOR and 68 FineWeb probes, reporting CP and within/between
   distances without pass thresholds. Reuse captured readings and budget their
   collection cost; the three-seed experiment is archived in FutureWork.
   Also log rows deleted per origin, value cut-off; luminosity of provisioned truths;
   the held-out two-truths
   probe and a fixed reconstruction sample; and from the concept-parts and
   native-perception landings: order-0
   inventory rows used against `nVectors`, provisional-pool occupancy and
   exhaustion warnings, raises per order with their run counts and stalled
   patience, refinement requests, lexical references left unknown per order.
   (The per-shared-operator gradient cosine and "persistent opposition"
   logging is withdrawn: each parameter has one objective owner since the XOR_grammar ownership change,
   so there is no shared operator to measure — [GradientFlow](GradientFlow.md);
   the dissonance diagnostic guards E against O
   on the maps instead.)
   And a resume test proving
   a mid-epoch checkpoint restores cursor, stream count, `refs` / surprise
   columns and forgetting counters with the next batch byte-identical —
   including the concept sidecar (feature groups, located brackets, sparse
   context, refinement buffers), the radix part groups and pending groups,
   the word-form index and the understanding's conceptual field, and the
   state added since: the address index and witness counts, the
   identity atoms and mints, the independent-components identity/change columns and
   relevance scales, the comparison reader's copy of the answer path
   and the math chain's pending relational rows. Exit:
   one command on a small config; the test in the suite.

### Mechanism residue, as it stood in todo.md

- **Mechanism residue of the closing mechanism and of occurrence, tense and
   surface form** (Alec, 2026-10-19: "even
   though we are on 4, 6 and 5.5 have incomplete work"; completed after
   the run harness, before the orders work). Mechanism work only; every learning *claim* stays at
   the full training session's gates. The copular c artifact belongs
   after stage 4e and the NP continuation; the paired `.0517` invariant
   is repaired in the preceding carried-invariants repair. Each line below lands or is moved to
   the full training session or FutureWork with the reason written here; when the list is
   empty those two entries collapse to Done lines.

   Native binding, MTM and participant menus are published at `cf61b5678`.
   Remaining root-participant semantics are [review follow-ups](FutureWork.md#root-participant-positions-review-follow-up-2026-10-21);
   the reference sites' grammar contract and reconstruction term are Done at reviewed
   source `3a9db635e`.
   Named token content before the close is Done at landing `e243d7bdf` (WikiOracle `2ffda78`), reviewed
   source `23a41540f`: the named preview token
   retains content, and its approved fixture correction passes every scoped
   gate.
   The store rename and cull are complete: separate rename `804d8adcb`, sole cull `0b030e3eb`
   (WikiOracle `baa2e7f`). Frozen cull source `d4f316411` has 223 passes,
   seven existing skips and Claude's approval. The storage-only change is Done at
   `39272e7a6` (WikiOracle `ade0399`), reviewed source `6a36b7cee`:
   1,144 scoped passes, nine existing skips, Claude verdict LAND.
   Tags, mention readings and the
   occurrence-only occupation ports await the derivation removal;
   identity threading comes first. No completion of the storage and identity
   step is claimed. All subsequent work stays on main.

   *The storage and identity step closed 2026-10-23* (identity threading,
   reading consolidation, one sweep; BasicModel
   `3624f6204`, WikiOracle `c9b19d4`). **The rest of the residue runs as two
   passes** (Alec, 2026-10-23): the constituent lessons, the verb and the modal, with the swap to
   operations on meaning and centroids on form brought forward from the orders work
   ("The whole swap now"), in the order swap, verb, constituent lessons rerun,
   modal; then the occurrence and surface-form remainder with context wholes
   and direct-operand word classes.
   The swap/verb freezes before the lesson rerun and the modal freezes
   after it; one combined review through Alec, gates sized to the change,
   one full sweep at the end.
   *First pass published 2026-10-24:* BasicModel `f306fc28aa`, WikiOracle
   `319a793`, Claude LAND:
   swap, verb and modal landed on the paired configurations; the constituent lessons'
   rerun stopped at description (1/4) and the modal corpus failed 0/4 in
   both forms, both carried to the full training session without rescue; six carried notes.
   The corpus's retained keep records show no alternative timeline was
   ever opened and neither teacher fired: the chooser never applied
   `adverb` to the marker, so the training session's retry starts from operator choice.
   The lessons, verb and modal therefore leave the residue at those landing hashes, with their
   learning remainder explicitly carried to the full training session. **The second pass is
   the surface-form remainder and the word classes only**, reviewed against the code; implementation is released.
   A short lesson smoke precedes each
   line's single freeze, then scoped gates and review through Alec.
   *Second pass landed 2026-10-24:* reviewed BasicModel `75345f891e`; published
   main `4f136f76faabafc24397765f4ffe90d01d3dc4bc`, WikiOracle
   `69f82f5f77a3980020841fe859a09afb7733e7be`, Claude LAND:
   covers and `.when` segments, centroids on form, VP phases as case
   selection, direct-operand occupations, the presentation boundary with
   persistent cover concepts (Alec), capacities 18/20/20 by derivation.
   Six placements 0/6 → the full training session (run before the boundary).
   History note (2026-10-25): history after `f306fc28aa` was rewritten before the push (`537523ae04`); `75345f891e`, `a34325c85d` and `f095569a1f` are respectively `2067dbc25a`, `3561f39aa0` and `31c3e4a59e` on origin.
   **The residue closes as a mechanism landing; its learning remainder goes to the full training session.**

   *From the closing mechanism*,
   described below by origin; the first-pass order above supersedes the older sequencing:
   - **Stage 4e, constituents — learning remainder moved to the full training session
     (2026-10-24):** on swap/verb source `65451694e`, original budgets
     pass source order 8/8 (both CE bars fail), lower and definite 1/1;
     description stops at answers 1/4 with brackets 4/4. No rescue.
     Adjective, definite-adjective, final evaluation and copular identity
     remain unrun.
     The retained learning contract is a passing gate
     (behaviors, no operator named): the next pass from the stage-3
     artifact with stage scoping of the menu and, **decided (Alec,
     2026-10-22), one action per operator**: the per-referent copies of a
     determiner leave the compose menu, the grammar chooses the
     operation, and the close's `bind` chooses the member.
     Grounding is textual; no perceptual
     grounding stage. Then the NP continuation: adjective and
     definite-adjective lessons, final evaluation, default sweep.
   - **Identity lessons** (a singular NP is a short-term concept, minted by recurrence in LTM1; matching at the close,
     revision); no native identity run before 4e passes. **`bind` as a
     thought operation (Alec, 2026-10-19, decided):** one operation named
     `bind` in the thought grammar, called once per referent, so the
     primary thought process sees binding as a step it chooses; its
     forward is the existing close-time resolver (`ReferenceClose`:
     native concept/occurrence candidates, follow-up credit and
     salience — the evidence still chooses whom), its reverse re-emits the
     description or determiner. **A site** (Claude's reading of the code,
     2026-10-19, confirmed by Alec) is any of three, none a word class:
     (1) an **open role** in the verb's frame — the verb (a group element) applied and
     an expected participant (the rebuildable cache of retained LTM participants)
     unfilled — which is the elided subject or object, with no word at
     all; (2) an operand that interprets to **no object of its own** (the
     pronoun: empty explanation, resolved by prediction,
     recency, role and the learned surface cue); (3) a **definite** phrase,
     today's only cue, kept as one source among three. The vestigial
     `ContextualBindLayer` was deleted when source (2) replaced its
     pronoun route. Called once per site; `bind` adds no scorer. The native-binding step
     replaces the shared resolver's features with two native meet ratios,
     participant prediction, prime, form-region containment, recency and
     the two roles; the cue matrix and identity dictionary are retired.
     **Concrete design:**
     declaration, executor over a `ReferenceClose.score` refactor, the
     three sites in code terms, with the native participant cache and
     `expected_roles` for elision, order of work, gates G1–G3; the vestigial binary `bind` layer in
     `Language.py` (left operand returned, reverse raises; the retired
     nearest-left rule)
     is deleted. This supersedes the pending choice on
     the chooser's reference variants: the variants collapse to the
     determiner mode on `lower`, and the individual is `bind`'s. One rule
     called per referent (minimality).
     **With it, the row amendment**
     (Alec 2026-10-19): three positions stay (argument, kind, argument;
     kinds `DEF`, *said*, *if-then*; arguments ideas or REFs); parthood
     and kinds are folds in the main tree, not rows; two forms of LTM named by
     their slots — **LTM1** (one concept per row, possibly unminted; ideas
     and their addresses; retention by surprise) and **LTM2** (two
     symbol references and the truth pair — one relation, `Part(x, y)`,
     read by its arguments' types; the same structure as the taxonomy over
     symbols; only minted concepts) —
     with STM as **STM1** — one wide concept vector per entry (the concept
     space's full width; computed in STM, never written to LTM, no
     inversion gate) plus the index of recent rows and the activation
     field; all of LTM2 in the native-binding step;
     the three signed-chart phase artifacts are retired and regenerated
     only when their lessons are next needed, strict rejection permanent; a concept is
     minted by recurrence in LTM1 (the access count, the type criterion
     again) and `bind`
     resolves references among minted individuals;
     the definitions index becomes a rebuildable cache over the DEF rows
     (verify forgetting clears it). **Row shape decided
     2026-10-20**:
     slots (1 or 3), `.when` as model time sinusoidally encoded and never
     read by composition, `truth_plus`/`truth_minus`, access count,
     `surprise`; the document and the stream become context wholes in
     WholeSpace; every other column and sidecar key is derivable, moved,
     cached or deleted per the table; LTM1 carries no identity.
     *The storage and identity step also carries (Alec, 2026-10-21):*
     one LTM store in strata: LTM0 rows (one concept, occurrences), LTM1
     relation rows `R` with two references to LTM0 objects (occurrence
     addresses or minted symbols; "S1 R S2"), LTM2 relations over LTM1
     rows; today's Part store merges in as the LTM1 stratum; a relation's
     argument needs no mint (recommended: the close-time mint ends,
     minting by recurrence only), and a referenced LTM0 row is protected
     from forgetting while referenced. Alec, 2026-10-21: the classes are
     renamed to match, `TernaryTruthStore` → `LTM0` and `PartStore` → `LTM1`,
     as the first commit of the store rename.
     *Sequence decided 2026-10-22 (Alec):*
     no sentence derivation in LTM, MTM stores working content,
     identity first. The **storage-only change** is Done at `39272e7a6`;
     **identity threading** is Done at landing `148953f9b` (reviewed source `65ab7fbe9`);
     **reading consolidation and the step's one full sweep** are Done: reviewed `fcaff2d19`,
     test-only sweep repair `9f5500870`.
     *Layout
     decided 2026-10-23:*
     LTM0 one slot (ideas), LTM1 two tagged references that chain, no
     sentence row; **moved to the orders work** (Alec, 2026-10-23) with STM's eight
     shared chunks.
     The storage and identity step here ends with identity threading, reading consolidation and the sweep.
   - **VP vectors by temporal difference — Done at `f306fc28aa`**
     (WikiOracle `319a793`). The per-identity
     teacher and verb-group action are landed; learning stays with the full training session.
     The following earlier formulation is retained as the decision record.
     *Formulation (Alec, 2026-10-19):*
     the **first difference** — the verb is a group element acting on
     the identity's frame (object before -> object after; a diagonal in a
     **shared basis per verb group**, groups learned by joint
     diagonalization and minted on recurrence — 2026-10-20;
     `<verbSpectrum>` is the one-group case);
     identity element = the undefined verb = unknown; support = the
     nonzero log-eigenvalues, zero by default; teacher = the coordinatewise
     log-ratio of the after- and before-frames read from the stream's
     occurrence rows; states are relations, not group elements (Claude's
     reading to confirm); and the **second difference** —
     modals and counterfactuals extend the event into possibility space, a
     whole of which the actual is a part — *de re* as membership evidence
     in the taxonomy, *de dicto* as an S located in a world that is a
     context whole (the minimal mechanism decided, defaults
     NP=everything / VP=exists (read "occurs") / MP=all worlds).
   - *Direction (Alec, 2026-10-21, tentative):* "probably verbs do too"
     become objects at their determination, "so we can bind those to help
     Modality": a verb's event, determined by its tense, aspect or modal
     operation, would be an MTM object bindable like a noun phrase.
   - *Direction (Alec, 2026-10-22, [Orders §10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower)):*
     modals are of the same order as verbs and nouns; "MPs are differences
     of VPs just as VPs are differences of NPs (not really differences, but
     diagonals within quadratics)"; "running and walking are both
     possibilities for Joe's movement"; treat them analogously to VPs.
     The modal is built as the verb is built: a group over
     a noun phrase's alternative verb phrases, a diagonal per modal in a
     shared basis. This answers the possible situation:
     an alternative VP, neither a minted world nor a lane.
   - **Modal mechanism — Done at `f306fc28aa`** (WikiOracle `319a793`;
     failed corpus learning moved to the full training session). *Superseded 2026-10-23
     (Alec):*
     a counterfactual is an alternative **timeline** of the one identity
     (MTM holds identities, connected NPs, and timelines, connected VPs on
     one identity); a modal clause is the ordinary composition with the
     lowering withheld, an abstraction above order 0, never an order-0
     occurrence; form is the `adverb` rule; minted worlds and `position`
     are withdrawn; continuation default and lifetime through promotion.
     The bullet below is the record.
   - **The second difference: the modal as the verb phrase's preposition
     over worlds** (Alec, 2026-10-19),
     after the tense pass: `position` called with a world as object,
     test part-of; worlds minted as individuals like the document; the
     *de re* reading is `S REL S` written as membership evidence in the
     taxonomy (`[Alec]` in `[running things PP(W)]`, a relative truth, a
     fold not a row) and the *de dicto* reading writes `S` (`[Alec runs]`,
     an LTM1 row whose concept carries the world as a context whole, an
     absolute truth) — Alec's two-truths syntax, 2026-10-19; the bracketing is learned by exposure; the teacher is the
     extension itself. No `MP` rule. Needs the world individual and
     nothing from the orders work. Gate: generated corpora with a modal marker as
     a free leaf and as a verb affix; both bracketings reachable; "did"
     vs "could" questions answered from the right store.
   - **Direct-operand word classes** — the distributional categorizer;
     output typing deferred.

   *From occurrence, tense and surface form* (its entry above):
   - *Checked 2026-10-22:* the legacy `timestamp` column is dropped at load
     and `Clause.eternal` is gone; nothing to retire. **Recency is store
     order, as landed** (Alec, 2026-10-22): `get_stm_chain` reads a stream's
     rows oldest first and re-witnessing keeps a row's place. When context
     wholes replace the stream column, preserve that store order; equal
     model ticks do not determine it.
   - The **learned surface transformations and their inverses** (absorb and transpose; split and insert left the grammar)
     — the old analyzer and tense/aspect/morphology layers are already
     absent. Verify the two remaining capabilities through paired native
     composition and inverse, trained by reconstruction under the existing
     step cost; remove stale language-naming text.
   - **Tense and aspect on VP phases** (the compound selecting
     among the verb phrase's phases; the preposition as one operation with
     two attachments), after the identity lessons above; the inert markers
     if tense is built on the preposition face. **Across languages**
     (Alec, 2026-10-19: "operators replace the table; if we don't have the
     operators to do that, our grammar is not universal"; "the operators
     should be minimal … just one rule called twice"): three
     rules — `position` (one rule, called once per selection, with or
     without a leaf; not called = whole = unknown), one marker-leaf kind
     (its reading = how `position` is called; nested wholes made at
     admission, not by a rule) and `surface` (absorb either side,
     transpose; nothing else) — and the claim that they reach every
     placement by exposure; **no language is named or special-cased anywhere in
     code, config, fixtures or tests** (the two "English" docstrings
     in `Models.py` go with this landing); gate = one grammar, one
     configuration, six generator variants of where the marker is
     written, no change to the mechanism; does not wait for the developed
     corpus.
   - The **nested-admission criterion** and its performance: the old
     shared-run probe carved 7 of the 17 4e words; Alec superseded the
     broader statistic with one named marker lesson, default off. Keep
     that decision and verify native paired use and reindexing cost.
   - *Specified 2026-10-24, third version:* context wholes:
     batch indexing on LTM and STM plus `.when` segment the store; a
     document starts with a cover row holding an automatically minted
     per-document concept in its batch's LTM stream; no run rows, no
     blend; a perceptual cover is FutureWork; the centroid is the center
     of parts and wholes, each averaged over its count.
   - **Documents (and streams) as context wholes** in WholeSpace, their trace in content (2026-10-20); learned before/after. *Added (Alec, 2026-10-22):* word adjacency is removed
     in the storage-only change with no replacement here: neighbours are distributional
     information, already carried into a word's meaning by the sentences it
     occurs in; a whole built from left/right neighbours "is not really
     characterizing the whole" (withdrawn the same day).
   - The certificates, measured: clock-independence,
     offset-independence, addressed-chain, interleaving, preposition and
     surface-inverse.

   - **Priming diffusion** (Alec, 2026-10-20): the first part (default
     `primingSpread` 0.05, diffusion along the one store's `Part(x, y)` edges
     and word-to-sentence membership) landed with native binding (`cf61b5678`);
     strata, order-0 geometry and placement belong to the orders work.
   Moved to the orders work (Claude, 2026-10-21, at Alec's invitation): order-2
   kinds, kind-level verbs and the generated math suite.
   Not here: dimensional `lift`/`lower` (FutureWork); the earlier learning
   gates and empirical evidence (the full training session). Exit: the
   list above empty.

### Priming as location-based diffusion, as it stood in todo.md

- **Priming as location-based diffusion in the one conceptual space.**
   *Decided (Alec, 2026-10-26).*
   Successor to Orders §9's stratified surface, which §10 withdrew. The
   placed meaning codes are the network: `Priming.propagate` applies the
   dot-product kernel to the stream's active rows once per sentence in the
   pre-composition `prime_seen` (decay, bump, propagate), stored and
   conservative; receivers within an adaptive radius, at most
   `geometricNeighbours`; class rows placed at the membership-weighted
   centroid of their members, valid while contiguous within that radius;
   `_priming_seen` ledger for placement; `diffuse`, `geometric` and
   `primingSpread` retire, `part_edges` stays as the membership cache. One
   pass: smoke, freeze, the gates, ports, the five
   scoped groups. Exit: the gates green on
   one hash, Architecture's priming guard reworded to the ledger.
   reviewed on b031b11ee9f2, approved, both document repairs applied.

### Order-0 concepts sorted over the perceptual mereology, as it stood in todo.md

- **Order-0 concepts strictly sorted over the perceptual mereology:
   two subspaces, one call.** *Decided (Alec, 2026-10-26;
   [Spaces](Spaces.md#the-cube-as-a-lattice-two-form-subspaces)).*
   The percept width splits at construction into an identity subspace
   (`WordIdentity`'s pair and length coordinates) and a property subspace;
   identity sorts by atom containment, property by extension over bytes
   (subtype smaller); location selects the set, never codes; a part's
   property subspace is the meet of the types over its bytes, a type's
   identity subspace is one; the concept form is the join of its parts and
   the meet of its types; `Ops.partOf` over the whole code is the one call;
   the interval rule and `_interval`'s upper bound retire. One pass: the
   layout split, the form rule, the certificate (tower 0 as respecified,
   tower 1 subtype pairs, cross-level at-most relations, nested order-0
   concepts strictly sorted; `containment_audit` becomes that check) and
   the context-wholes form ports. Independent of location-based priming. Exit: the certificate
   green on one hash.
   **Built (2026-10-27), unlanded:** the certificate and native lesson pass
   on `be4514dedea8d`; Alec's governing partial-order clarification: no required
   part/type inequality; equal-code ties allowed.
   **Gate blocked:** the core native worked-document test fails in thought
   persistence; all other scoped groups complete without failures. Slow
   cases and old-width snapshot regeneration remain deferred to November.
   Return through Alec for review before another revision.
   **Latest freeze, gate-complete (2026-10-27), unlanded:** `555e63e2f521`;
   both demonstrated boundary defects repaired; all requested scoped
   groups pass, both former core failures pass, slow cases deferred.
   Return through Alec for Claude's review; no commit, push or WikiOracle bump.
   **Landing approved (2026-10-27):** reviewed and approved on
   `555e63e2f521`. Deferred slow cases and D5's snapshot regeneration
   remain scheduled for the November sweep.

### Corpus at target size, as it stood in todo.md

- **Corpus at target size.** **Closed code-complete (2026-10-29):** review freeze
   `472f5000850e`, reviewed.
   Lesson-13 completed 160/160: ideas pass (two distinct order-1 objects),
   relations fail (no shared order-2 kind), answers **0/80 with text**.
   Final curriculum completed **131/304**: proper phase passes, mass
   phase fails, count and kind-verb phases not reached. These are the
   measured results; code completion does not claim acquisition or the
   November landing. The thirteen scoped groups at the review freeze
   passed (2,545 passes, 57 skips); no tests or `RUN_SLOW` rerun for this
   close.
   **Carried work and owners:**

   - **The November landing:** capacity search and the run
     on this computer, D5's snapshot regeneration, the slow suite and
     one full sweep. Before those runs: repair the supplied `yes`
     admission defect; build D7 (the inverse's output as the correction);
     capture and repair the rejected MTM identity value, the non-finite
     owned meaning-kernel gradient, the final curriculum's unavailable
     query concept reference at sentence 132, and the earlier
     order/parts disagreement. These producer failures remain unresolved.
   - **Thinking:** D8, graded binding cost plus text cost so a
     better binding can win and credit the chooser. Done (landed,
     `3718d9d0dc`).
   - **The full training session's learning remainders:** the unforced 1-to-2 raise and
     the mass, count and kind-verb measurements, with their budgets.
   - **The next change opening `WordIdentity`:** atom property fill.

   **Earlier implementation record (superseded by the closing status above):**
   Corpus assembly, stage-preserving packing and native checkpoint continuation
   are built; 157 scoped checks pass after the capacity, centroid and serial
   discovery repairs. The 12-sentence native diagnostic now runs the serial
   observer and passes identity checks, but acquires no named objects or classes.
   The full target constructs within a 28 GiB guard; training memory and
   the separate relation/witness capacities still need provisioning. The
   completed-understanding inverse has a native mechanism certificate.
   Shared initial class learning, exactness of the remaining approximate
   paths and unforced acquisition remain open; no completion or landing is claimed.
   **Familiar tier (2026-10-28):** built on frozen
   source `4b83dec0bd0e` (938 files), including shared initial class learning.
   All scoped groups passed: 2,236 passing executions and 44 skips across
   overlapping groups, zero failures. The native lesson completed all 48
   sentences, four presentations per stage, with clean form identity audits;
   acquisition failed: neither proper name acquired an object, and `cat`
   did not acquire the shared order-2 kind. The existing presentation boundary
   clears MTM between repetitions.
   Return through Alec for Claude's review; no commit or publication.
   **Reading and epoch (2026-10-28):** repairs and
   supplied-answer lesson-13 are built. Freeze `b84b983d2ddc` preserved the first
   inverse performance repair; its partial lesson was stopped for the additional
   repair. Freeze `103c66621c4c` passed 81 acceptance checks and 2,388 scoped
   executions (53 skips). Its lesson was paused at 44/160 sentences and then
   terminated for the generation repair.
   A native 144-step walk is exactly preserved at 9.74 s before / 0.62 s
   after. The separate c/e sample is 19/20
   against baseline 20/20. It and the earlier introduction failure remain
   review issues.
   **Generation repair:** frozen at `aa1b32288f71`; the unary-chain
   bound, completion-first comparisons, winning-departure credit and creation-time
   identity order are built. All twelve groups pass: 2,421 passes, 55 skips,
   zero failures; the native smoke passes. The fresh lesson completes 37/160
   sentences (median 11.77 s) then fails at sentence 38 with a dangling local
   constituent reference in a supplied query result. Relations were not reached;
   the cube failure did not recur and remains carried. The 48-sentence diagnostic
   did not reproduce the reference error; the candidate-table-loss probe is a
   lead, not a confirmed native cause. Returned through Alec for Claude's review;
   no commit or publication.
   **Reading census (2026-10-29):** source `765e7667a680`
   repairs the candidate binding/table defect; the requested diagnostic finds no
   MTM candidate scan/charge defect. All twelve groups pass (2,423 passes,
   55 skips), and lesson-13 completes 160/160 sentences without a fill or
   cube crash. Acquisition fails: Felix gains an order-1 object, Ciao does
   not, no shared order-2 kind forms, and answers score 0/80. The census
   records 3,306 of 3,520 walks ending at the unary bound and zero departure
   wins; the supplied yes is absent from all 576 applicable terminal banks.
   **Final design proposed (2026-10-29):** generation,
   two-lane DEF learning, percept recombination/inversion, noun/kind
   measurements, guarded B28/W256 capacity search and corpus run, then the
   November snapshots/slow cases and landing sweep. **F1–F3 approved by
   Alec (2026-10-29):** F3 is percept expectation, trained in the
   processing loop as percepts arrive, with one local update per external
   arrival and a frozen prior shared by both trials. Full content and the
   signed residual remain separately available. Claude approved the design
   with A1–A4. **Final implementation (2026-10-29):**
   F1–F3 and the percept-only residual split are built in the live tree;
   source verification, scoped gates, lesson measurements and guarded capacity
   work are in progress.
   No acceptance freeze or landing is claimed. The conceptual residual half
   remains with the thinking work; D5, slow cases and the landing sweep await November.
   **Alec's deferral (2026-10-29):** large-file B28/W256 capacity search and
   selected-prefix training are registered in `test/test_corpus_large_slow.py`
   for November; the partial search has no final fit. Scoped gates and the
   lesson curricula continue now, with `RUN_SLOW` unset.
   **Review freeze `ccc5e461740f` (2026-10-29):** 956 files;
   2,515 scoped passes, 57 skips, two core failures. Lesson-13 completed
   160/160: ideas pass, relations fail, answers 0/80; no acquisition artifact.
   The final curriculum stopped at sentence 15 with an unwritten-introduction
   bind. The residual split's current chart conversion retains the full reconstruction
   gradient and needs explicit review. These remain open; return through Alec
   for Claude, with no landing claim.
   **Producer repairs:** freeze `0be76c6f84f4` passes all thirteen
   scoped groups (2,541 passes, 57 skips). Lesson-13 completes all 160
   sentences: ideas pass, relations fail, text available for all 80 answers
   but exact accuracy 0/80. Final curriculum exposes two further failures;
   the native-symbol origin repair and epoch capture continue.
   The correction-output form of the residual split awaits Alec's D7
   confirmation and is unbuilt.
   **Review handback `472f5000850e`:** all thirteen scoped
   groups pass (2,545 passes, 57 skips), with native smoke and allocation
   checks. Diagnostic lesson-13 completes 160/160: ideas pass, relations
   fail, answers 0/80 with text present. Final curriculum stops at sentence
   132 with an unavailable query concept reference; earlier MTM-value and
   owned-gradient failures remain unresolved. D7 is pending and unbuilt;
   return through Alec, no landing claim.
   **Clarified 2026-10-27 (Alec):** expectation predicts at each level;
   the approximately inverted object is the fully processed grammatical
   understanding.
   *Earlier scope (ownership now follows the carried work above):* the unforced 1-to-2 raise certificate; the
   fold trained from DEF rows with explicit and implicit definitions judged
   in the two lanes; parts of speech learned in order (proper, mass, count)
   and measured, not forced; order-2 kinds and kind-level verbs; the
   generated math suite under `RUN_SLOW`; the invertible-path program and
   the content width audit; the capacity, load and run-config measurements
   of the Exit below; the age-ordered corpus assembled; and, from the mechanism
   residue's list, the paired native surface route and counted centroid inverse
   certified for absorb and transpose. The learned-orders ledger
   inside is history. *Moved from FutureWork (Alec,
   2026-10-21):* **orders are learned, not written into the grammar**
   (design record below).
   **Carried from the learned-orders landing (2026-10-26):** certify the 1-to-2 raise on unforced text, “Felix is a cat. Ciao is a cat.”, measuring the shared word's object and both members without `force_order`.
   Alec: "we need not create an ordered grammar, because we have the
   rule that raises order." Parts of speech stay order-free.
   - **The definition: IS and its raise.** An equality writes an LTM1
     row and changes no order. When a second, distinct member joins
     the same side, the shared referent is raised one order and the
     word is given the raised concept as its object. "The same side"
     is the symbol the mentions read to, through their reading rows.
     Distinct means two events at different places or times, or two
     different identities in MTM, so "y is three. x is three." raises
     nothing. The trigger is the second member joining the singleton
     the equality's close marks (`PartClosing` ≈325);
     `ConceptIndex.index_part_row` performs the raise, which has no
     live caller today and must wait for that second row. This creates
     individuals ("this is Felix, that is Felix": 0 to 1) and kinds
     ("Felix is a cat, Ciao is a cat": 1 to 2).
   - **Return to S0 by lowering** (amended 2026-10-22, [Orders §10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower)).
     One composition lowers one order: NP + VP, predication. The modal
     does not (Alec, 2026-10-23, reversing 2026-10-22): it withholds the
     lowering, so a modal clause stays above order 0 as an abstraction on
     an alternative timeline of its identity.
     A determiner does not: it selects one
     of a plurality, and the phrase's order is that of what it selects,
     by lookup (a member of a kind, or a shaped portion of a mass, is an
     individual; "the water" is order 1). Order assignment is by lookup
     throughout: `lift` and `lower` keep their sigma and pi folds, and the
     stamp machinery (`ReferenceContext.operation_order`, `result_order`,
     `LiftLayer.order_delta`) gives way to reading the denoted concept's
     order, set by the equivalence classes. The mechanism residue builds the modal
     itself; its lowering is this line's. Raises carry no price; binds
     are credited through the derivative, since MTM feeds output
     prediction. The comments at `Language.py` ≈627 and ≈1577 still say
     `lift` raises order.
   - **Every symbol's order; folds with definitions** (Alec, 2026-10-21;
     design record below).
     Every concept has an order, but 0 doubles as "unknown", a recorded
     order overrides structure unchecked, relation rows always read 0,
     and a test-only API keeps an outdated ladder. Alec: keep both the
     explicit and the implicit definitions. Several DEF rows in LTM1
     that share a side train the fold the symbolic order loop uses,
     and "their match or mismatch is a judgement process, just as in
     human cognition", carried in the two truth lanes. Both raises
     still need two distinct members, with thresholds by strength of
     evidence. The fold raise runs only in `XOR_exact.xml` today.
   - **Learning parts of speech** follows the same order: proper
     nouns, then mass nouns, then count nouns where the language has
     them. Until this lands, the mechanism residue's order-1 stand-ins stay, and unit
     tests force parts of speech.
   - **Certainty** (Alec, 2026-10-24; "if the only work is the certainty
     work, let's act on it in item 3"): an order-0 concept is a point on the
     hypersurface with one scalar certainty; a higher-order representation
     is a region containing its members' points; certainty is orthogonal to
     the positive and negative evidence (Alec), which stay the row's two
     poles. Claude's reading of certainty, to confirm: the determinacy of
     the content, how sharply the point is located, which grows with
     witnessing (today `defined_code` over `witness_count`), one scalar per
     point, never the evidence magnitude. *Alec, later: the mixture in
     today's readers is right:* `Queries.extent` and `gain` read a lane's
     norm, which is definedness times evidence, and "values of one are
     well-defined and certainly categorized"; a support of 1 means the
     concept is learned and the categorization certain. So no separation
     is built; this bullet is the design record of the geometry, and the
     only work left under it here is an extent for higher-order regions
     if that is ever asked for. Centroids stay the symbol side's; a
     concept's meaning stays placed by context. *The geometry (Alec, 2026-10-24):* "the concept
     vector magnitude approaches 1 as we learn it; the magnitude of the
     candidate object has evidence, and the sum of the positive and
     negative evidence determine its length; the dot product is then 1 if
     we have only positive evidence, and are certain of the definition of
     the concept." Certainty is the concept's magnitude, evidence the
     candidate's magnitude split over the two orthogonal lane subspaces,
     and the two multiply in the dot product; negative evidence lies in the
     against lane, orthogonal to a positively defined concept, so it
     lengthens the candidate without entering the dot (it lowers the
     cosine). Audit of today's length and sphere readers in
     [FutureWork](FutureWork.md#concepts-as-regions-certainty-from-evidence-no-sphere-alec-2026-10-24).
   - **Layout and removal** (Alec, 2026-10-23: "the layout spec should be
     done with 3.0"): LTM0 one slot (ideas), LTM1 two tagged references
     that chain, no sentence row, the stored derivation and its readers
     removed, `AnswerProgram` renamed `Reading`; then STM as eight shared
     chunks. Lands with placement from the priming surface and the
     geometric channel, which it needs; it also removes the per-forward
     scans of the whole store (placement's context means, priming's
     membership pairs). The layout step is approved; Alec commits source SHA256
     `34848e154f0fa9912223d4117136d7eb63657cefaeca94a76c6574559a73819e`; the interim
     context source and remaining scans
     are addressed by placement,
     approved by Claude; Alec commits source SHA256
     `1e94238b57c99ba2b1003de2467e6f6acf543b68dc425263bed708509677d55e`:
     **1,003 passes and seven standing prerequisite skips under the documented
     protocol**.
     **The third step's repair is approved**, source `4b7fa2aaf7`.
     Both closing items are implemented: placement's unchanged replay has
     greedy coverage 1.0 too, carrying the 6/8 companion residual to the full training session;
     estimates carry zero time, keep current time as input and consume no
     arrival tick. Closing source `cd49a1e845` passes **1,113 cases with seven
     standing prerequisite skips and zero test failures**, plus the smoke.
     The fourth step's repair `acac0b3448` is approved as mechanism work. The approved
     two-stream port closes gate 6: the file and both decomposition arms
     pass 6/6, companions 6/8.
     **Orders learned**, the fifth step, is approved on `c4a1d92e3f`:
     1,647 passed, ten existing skips, zero failures, plus the smoke;
     companions remain 6/8.
     **Learned orders LANDED — 2026-10-26**, approved. The landing names orders learned `c4a1d92e3f`, serializer `caece43fc0`, sweep repair `03fbbf2a7d9c`, final model repair `6fe273d2f12a7`, and test budgets `ee7ca4e920bc`. The learning records stay with the full training session at their listed budgets; the four runtime failure groups move to the compiler work as named below; the unforced 1-to-2 raise certificate for “Felix is a cat. Ciao is a cat.” moves to the corpus at target size.
     *Decided later the same day (Alec):* every mental event
     has a perceptual and a conceptual part; episodic memory is the
     percept part of STM and LTM, semantic memory the concept part;
     `.when` belongs to the percept part, so `LTM0.when` and `LTM1.when`
     retire into the band and recall by time is a cue with only the
     percept `.when` filled; an order-0 concept is an event with `.where`
     and `.when`, higher orders abstract them; a sentence's percept part
     is its surface (the form block's fold over its words, the retained
     wording), stamped at the sentence level only; STM rows have the
     LTM0 form; STM is eight shared chunks with least-recently-read
     eviction off the stack; the object supplies its own Q basis for
     transitive verbs (the fourth step).
     *Decided (Alec, 2026-10-23):* **operations on meaning, centroids on
     form.** Today the learned kernels act on the form block and headed
     operations pass one operand's meaning through ("Felix runs" is
     Felix's meaning). In this work every grammar and logical operation acts
     on the meaning block, logical ones by their exact rules and
     grammatical ones by learned kernels; the percept part of a reduce is
     the centroid of its constituents' forms; zeroth-order concepts start
     random and are moved by the priming surface; generation reverses
     meaning composition and reaches forms through the symbol index.
     *Moved to the mechanism residue the same day* (Alec: "The whole swap now"): built in
     its first remaining pass.
   - **Retire mode** (Alec, 2026-10-22: "Removing mode should be a part
     of 3.0"; moved from FutureWork).

     Alec: "The model produces output when it wants to, and it calls ask() or
     query() when it wants." Inside a thought episode that is already so: the
     chooser picks query, conclude or output. What remains is whether an episode
     opens at all. `ThoughtReferences.needs_episode` opens one for a closed
     sentence with an open slot ("what is x?"), and otherwise only when its live
     mode is interrogative and its evidence empty, which is how a yes/no
     question ("is x three?") gets one. Without mode, that question looks
     exactly like an unasserted "x is three": no open slot, no evidence. The storage step
     stops storing mode;
     retiring the live mode means offering the episode at every close, with not
     thinking among the chooser's options, so the decision is learned from the
     parse's form. The remaining live uses go with it: `question()` in
     `ThoughtClosing`, the record kind `question` set at write, the asked
     roles' mask in the expectation image (`Models` ≈20276), and the answer
     seed that skips interrogative fields (`Models` ≈8700).

     Alec: "The model always gets a chance to say something. But it should
     learn to say nothing unless it is asked a question." The first half
     already holds: an output is produced for every row. The second does not:
     the answer loss masks out rows with no supplied answer (`Models` ≈14193,
     `_what_answer_target`), so silence is never taught. The teacher, which
     knows from the data where an answer is expected, gives silence as the
     target on every other row, and the model learns from the form when to
     speak. Its gate is a learning claim, measured at the full training session.

   - **Design record: order in the parts of speech** (moved from
     FutureWork, 2026-10-21).

     Alec: "Would the grammar benefit from order? … how do we learn that some
     nouns are categorically different than others in a cognitive sense, and
     live at a higher order? A collapse to NP is, in some real sense, not
     correct." Moved here from FutureWork (Alec, 2026-10-21). The
     paragraphs below are the design record of the lines above, in the
     order they were written; later paragraphs supersede earlier ones.

     **The gap.** Parts of speech are recorded as occupations of operand
     positions, with no order (`PartsOfSpeech.observe_kept`), so every noun
     falls in one category. Participation cannot separate "cat" from
     "species": both take determiners. Only order relations across a sentence
     can: in "the cat is a species", species sits one order above the cat.
     Kind-level verbs such as "die out" or "spread" likewise take
     only kinds as subjects, which a flat category cannot say.

     **Two order-1 stand-ins in the mechanism residue.** Without order in the parts of
     speech, it works with two assumptions:

     - A determiner always outputs order 1 (`ReferenceContext.operation_order`),
       whatever its noun, and never raises the noun. "The cat" comes out
       right. "The explosion" should pick out one event at order 0, and
       "a species" one kind at order 2; both come out as order-1 individuals.
     - A bare word is order 0, a type standing for itself, until something
       says otherwise. A bare name such as Felix should be order 1. The parse
       cannot tell a name from an event word, so the fixtures supply the
       name's order.

     **History.** The May 2026 order-typed grammar (git `6c92c790c`)
     declared `NP3`, `NP4`, `VP1` and `S4 = lift(NP3, VP1)`, with polymorphic
     forms `NP*`, `NP*+1` and `NP*-1`; lift raised order and lower lowered
     it. The grammar was later collapsed to role-only positions, with
     categories recovered from participation ([Language](Language.md#grammar)).
     `Grammar._parse_category` still accepts the old forms, and
     `test_role_collapsed_grammar` bans them from shipped grammars. That scale
     also differs from [Orders](Orders.md): lift now keeps its subject's
     order.

     **Claude's recommendation.**

     - *Relations, not constants.* Each rule states how its operands' orders
       relate: lowering one down, membership one up, equality the same. Fixed
       labels such as `NP3` would declare a word's order in advance, a lexical
       fact the grammar-only contract excludes.
     - *Inferred, not filtered.* The parse solves for orders as type
       inference does, and a word whose position requires a higher order is
       raised through the existing raise (`ConceptIndex.index_part_row`). The
       chooser does not filter by order, so nothing is blocked while every
       word starts at order 0.
     - *Parts of speech carry it.* Occupations are recorded by position and
       order, so a noun at order 1 and a noun at order 2 are different parts
       of speech, learned from participation.

     This replaces both stand-ins: a name becomes a noun at order 1 by its
     occupations, and a determiner's output follows its noun's order. It
     changes the parser, the occupation records and the determiner fixtures,
     so it belongs with the order-2 kinds and kind-level verbs, not in
     the mechanism residue.

     **Learning parts of speech (Alec, 2026-10-21).** "We need to learn parts
     of speech. Much of that work will be in virtue of learning certain words
     first (proper nouns, then nouns (mass and non-count), then count nouns
     (for languages that have them)." Claude's reading: this order follows
     the orders. Names are individuals at order 1 and need no determiner.
     Mass nouns are types at order 0. Count nouns are kinds at order 2, which
     a determiner lowers. Each stage therefore gives the parse one new order
     class to infer. Stages 4a and 4b of the
     [curriculum](Training.md#the-generated-supervised-curriculum-in-order)
     are where this order would go. Until parts of speech are learned, unit
     tests force them. Alec: "Unit tests can force parts of speech; they may
     take a while to learn, and we need to test without that."

     **Historical marker occupations (Claude's review of the reference sites, 2026-10-21).**
     The grammar-only contract is implemented as specified: any recorded
     `surface_argin1` occupation makes that symbol a literal in implicit
     participant reads and a marker in `PartRead.argument_reader`. This is a
     history-based classification, not a check of the present occurrence's
     position. Revisit that distinction with parts-of-speech learning. The
     `ReferenceContext` comment still calls it an "objectless marker"; update
     that wording with the future classification work. This is distinct from
     the root-position and re-identification findings. No source change or
     new learning claim is made.

     **The learning sequence (Alec, 2026-10-21).** "This is Felix, that is
     Felix: 0 → 1. Felix is a cat, Ciao is a cat: 1 → 2." Demonstratives
     ("this", "that") point at events, at order 0. Alec: "'Is' does not
     cross order by itself, but when used with multiple identical RHS or
     LHS, it creates an equivalence class." The class is one order above its
     members: the events this and that make Felix, and Felix and Ciao make
     cat. This is Frege's definition by abstraction, as a direction is the
     class of lines parallel to a given line. Since no one asserts "a is a",
     nearly every definition adds a second member to some class. "Paris is
     the capital of France" is the identity case: both sides pick out one
     individual, and the definite article allows no second subject, so no
     class forms and nothing is raised. A determiner's lowering picks a
     member out of such a class. In the code today, binding occurrences into one identity yields
     an order-1 identity concept, but the word itself stays at order 0. The
     raise of a parent above its part (`ConceptIndex.index_part_row`) has no
     live caller, so closing "Felix is a cat" does not raise cat yet.

     **Where the two sentences are stored (open; decide with this entry).**
     Alec: "in the two sentences I gave, we have an S1 and an S2, possibly
     (which would be stored in LTM1 and LTM2, I guess?)". As decided for the storage step, strata are
     structural: a relation over LTM0 objects is LTM1, and one whose argument
     is a relation row is LTM2. Both sentences are therefore LTM1 rows, and
     their orders 1 and 2 belong to Felix and cat, in the conceptual order.
     Alec's reading would put each relation in the stratum of the class it
     adds to, so "Felix is a cat" would be LTM2. LTM2 is derived when read, so
     this changes how rows are read, not how they are stored, and the storage step
     proceeds as decided. In either reading, the sentence as read is an
     order-0 occurrence in LTM0, linked to its content by the reading row;
     closing a sentence changes no order.

     **Learned orders instead of an ordered grammar (Alec, 2026-10-21).** Alec:
     if UG learns the ordered grammar, none need be written, "because we have
     the rule that raises order." Claude checked this against the code the same
     day. The operations already express the ordered rules: lift, verb and the
     relations keep order, and equality is stored as membership in a singleton,
     changing no order. Three pieces are missing before the orders are learned:

     1. **The raise is not wired.** `ConceptIndex.index_part_row` is called
        only by tests, the close's `_index_part_row` hook is never read, and a
        grammar-requested order only selects an existing concept
        (`ConceptualSpace.resolve_word_concept`); it never creates a raised one.
     2. **Nothing forms a class from repeated "is".** Equality rows that share
        a side are never grouped, and a word's object is set only inside
        `Interpret`, never from a closed sentence. After "this is Felix, that is
        Felix", Felix still reads at order 0.
     3. **The determiner's output is fixed at order 1**
        (`ReferenceContext.operation_order`). It can follow its noun's order
        only after nouns are raised, since every word starts at order 0.

     The raise needs one condition: the shared side must hold two different
     identities, or two different events. Otherwise "y is three. x is three."
     would raise three, though x and y are both bound to three itself. With
     these three pieces, parts of speech can stay order-free, because the order
     lives in the concepts. Claude's earlier recommendation to record
     occupations by position and order would then be unnecessary.

     **One raise, three lowerings (Alec, 2026-10-21).** "So we need IS and it
     needs to raise the order when necessary. And two things lower the order:
     a determiner and the combination of NP and VP." He added the third, "vp
     mp for modal". Why predication lowers
     (Claude): a noun phrase names a class at order n, the invariant across
     its members, as an individual is the class of its occurrences. A verb is
     the step between two of those members (the first difference), so it
     holds at some members, not at the class as a whole.
     "Felix runs" therefore picks out occurrences of Felix, at order 0, one
     below Felix. A kind-level verb does the same one order up: "dodos died
     out" is a change across the kind's members, at order 1. At order 0 the
     step stops: a mass noun's instances are located portions, still order 0
     ("water flows"). The determiner lowers for the same reason, picking a
     member out of a class, and IS raises, collecting members into one. The
     modal lowers the same way: it ranges over possible worlds, so a verb
     phrase under it holds in some of them, not all, and the combination
     lands among those possibilities, one order down. In the
     code, `LiftLayer.order_delta` would become one down, stopping at 0. That
     leaves sentences about individuals at order 0, as now, and changes only
     sentences whose subject is a kind.

     **Creating individuals by the same rule (Alec, 2026-10-21).** Alec: "def
     will create LTM1 that, once there are multiple LTM1 that have the same
     side, raise the order of the referent LTM0. Might this work even for
     creation of Felix in the first place?" Claude: yes. "This is Felix" writes
     one LTM1 row from the event "this" to Felix, and nothing is raised.
     "That is Felix" writes a second row with the same right side and a
     different event, so Felix is raised to order 1: the class of those events,
     which is what an individual is. Hooks in the code: an equality's close
     already marks its right side a singleton (`PartClosing`, ≈325), so the
     raise fires when a second, distinct member joins that singleton.
     `ConceptIndex.index_part_row` already performs the creation, making a
     concept one order up that inherits the name's forms and holds the member;
     it raises on its first call, so the trigger must wait for the second row.
     Three conditions are new:

     1. "The same side" is read through each mention's reading row, since
        every mention of Felix is its own LTM0 occurrence; what the rows share
        is the symbol those mentions read to.
     2. The other sides must be distinct: two events at different places or
        times, or two different identities in MTM.
     3. The word must be given the raised concept as its object. Today a
        word's object is set only inside `Interpret`, so "Felix" would keep
        reading to itself at order 0.

     In the mechanism residue, Felix is created differently: with its order forced to 1, its
     first mention becomes its own identity in MTM, later mentions bind to it,
     and identity threading promotes the identity after three sentences. This rule comes
     before that, as the way a name reaches order 1 at all; binding and
     promotion then proceed unchanged.

     **Every symbol's order; folds and definitions (Alec, 2026-10-21).** Alec:
     "please verify that all symbols have an identifiable order, and let's
     coordinate the use of folds to increase order and the use of definition;
     it may amount to implicit vs explicit learning, but we need to make sure
     they play nicely with one another." Claude checked the code the same day.

     *Orders.* Every concept has an order from `ConceptAllocator.order_of`: a
     recorded order first, else one above its highest symbolic part, else 0.
     Words record 0 at admission, identities 1, and a raised taxonomy parent one
     above its member. LTM rows carry their own order column, read when a row is
     not an allocator concept. An unnamed learned row takes the order of its
     storage slice (`ConceptualSpace._row_order`). MTM tokens record no order
     but hold only individuals. Four gaps:

     1. 0 is also the default, so "order 0" and "never identified" read the
        same; a new name and an event type cannot be told apart.
     2. A recorded order overrides structure unchecked. One disagreement exists
        today: a raised symbol over raw codes reads 0 from the allocator and 1
        from `ThoughtConceptualCapability.order`, which takes the larger of its
        order and its raised flag.
     3. A relation row's handle has no parts, so every LTM1 relation reads
        order 0 as a symbol, whatever it relates.
     4. `conceptualize`, called only by its own test, keeps an outdated ladder
        (order 1 "object isa word", order 2 "higher-order object").

     *Folds and definitions.* The fold raise (`ConceptualSpace.maybe_raise_order`,
     in the promotion pass, enabled only in `XOR_exact.xml`) gives two rows of
     one order that occur in matching contexts a parent row one order up,
     recorded as weighted edges between rows, which `order_of` does not read.
     The definition raise (`ConceptIndex.index_part_row`, unwired) writes a Part
     edge and a recorded order. So one class could be made twice, with its order
     read two ways. Orders §9 already decided the remedy: above order 0 nothing
     is stored but `Part(member, whole)` edges, and centroids are derived
     caches. Claude's proposal:

     - *One membership record.* Both raises add the class's members as Part
       edges, so one reader derives every order; the fold's row weights become a
       cache over that membership.
     - *One class per member set.* A definition whose members already form a
       fold class names that class instead of making a second one, and a fold
       over members of a named class joins it. Implicit learning finds classes,
       explicit learning names them, and they meet when a word is given a class
       already found.
     - *One distinctness rule.* Both need two different members. The fold raise
       already requires the matched row to be absent from the current field.
     - *Thresholds by strength of evidence.* A definition raises at its second
       distinct member. The fold raise keeps its context match and, above
       order 0, its stalled-refinement gate.

     **Integration by training (Alec, 2026-10-21).** "I think that multiple
     DEF in LTM1 can be used to train the fold being used in the symbolic
     order loop, and that's really all we need for integration. We can
     keep the explicit and implicit definitions, and their match or
     mismatch is a judgement process, just as in human cognition." This
     supersedes the proposal above of one membership record and one class
     per member set; its distinctness rule and thresholds stand. Claude's
     reading: a class built from DEF rows supplies the fold's target, so
     the fold over those members should reach that class. The judgement
     needs no new machinery. A definition is evidence for membership and
     the fold's own reading is evidence too; agreement fills one lane,
     and a mismatch fills both, the corner already decided to prompt
     refinement (Alec, 2026-09-23). The lanes are never netted.

     **Clarified 2026-10-27 (Alec):** raised classes need initial learning
     to cover their members; explicit and implicit classes receive the
     same treatment.
     Synchronizing explicit and implicit meanings is a separate
     [future-work item](FutureWork.md#synchronize-explicit-and-implicit-meanings-alec-2026-10-27).

   *Added 2026-10-20 (Alec):* placement from
   the registered priming surface — SBOW's context is the surface after
   the seen bumps and the symbolic diffusion, before the conceptual
   channel; the objective in the cube's own metric (raise the meet ratio
   with the registered context, lower it against negatives; the collapse
   diagnostic as the gate); one metric for SBOW, diffusion and `bind`
   ([FutureWork](FutureWork.md#placement-from-the-registered-priming-surface-one-metric-for-sbow-diffusion-and-bind-alec-2026-10-20-proposed-with-the-objective-to-decide)).
   *Reversed 2026-10-22 (Alec; [Orders §10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower)): one conceptual
   space, concrete and abstract intermingled; the per-order spaces and the
   stratified surface read below are withdrawn, while the symbol tree, the
   Part edges as membership, membership by edge and the cross-order
   diffusion stand. The record:*
   *Added 2026-10-20 (Alec; [Orders §9](Orders.md#9-all-symbols-in-one-mereological-tree)):*
   **per-order conceptual spaces, ramsified over the one symbol tree** —
   order *n*'s space is the cube over the order-(*n−1*) symbol index, its
   concepts characteristic vectors realized by LTM2's Part edges, similarity
   the meet of membership vectors, centroids above order 0 only as
   validated caches, cross-order comparison by membership; and **the
   stratified priming surface** — one vector read per order (a point in
   every order's space at once), the kernel edge-based across and above
   order 0 and geometric only at order 0, read points declared per
   consumer (placement: the order-0 stratum before the geometric channel;
   `bind`: the request's order; attention and the chooser: all strata).
   *Added 2026-10-21 (Alec):* **`intuit`, one layer of order-0
   prediction in the parallel calculus**:
   union, intersection and negation (lane exchange) as one max–min layer
   over remembered `S1 → S2` LTM1 rows (Mamdani: fire at the meet ratio,
   union of clipped consequents; no parameters; after the storage step); it leaves the knowing set, from
   which the answer and output are composed; expectation stays the
   bottom-up gain and its three top-down reads move to `intuit`.
   *Moved from the mechanism residue (Claude, 2026-10-21, at Alec's invitation):*
   **order-2 kinds** and **kind-level verbs**
   (the verb groups acting on a kind's extension over time, such as
   "die out" or "spread"), both on the per-order spaces above; and
   **The generated math suite under `RUN_SLOW`** (Alec, 2026-10-20): a
   generated, graded corpus of mathematical sentences in English that
   trains a grammar for real, at scale, with the learning claims as its
   gates (names and equality, roles, decomposition, `what` bound,
   conditionals and counterfactuals, ellipsis, generics, attribution,
   the inverse verb and the verb groups' exact bases, the carrier
   variants) — [Testing](Testing.md#fast-regression-and-slow-experiments),
   coverage in [Training](Training.md#the-generated-math-suite-what-it-exercises-and-what-it-cannot).
   The same run is the compiler work's parity test at scale (compiled and
   eager agree in actions, probabilities, costs, updates). Built from the
   `MM_math_chain` generator in the Viterbi order; throughput recorded,
   since the compiled runtime is where it bites.
   *Takes from the expectation mechanism (Alec, 2026-10-19):*
   the invertible-path program — exact inverted targets for the maps,
   sources going live, reconstruction retiring to an audit where the
   inverse is exact, the width `D ≈ c·k·log N` for the audit to read zero
   ([FutureWork](FutureWork.md#expectation-by-inversion-and-the-reconstruction-audit-decided-2026-10-06-unscheduled)).
   Expectation no longer depends on it; the full training session's condition reads it from
   here. Raise `maxDocs`, exercise multi-shard if needed,
   measure sentence-list/address-table memory and loader time, confirm
   `resume_skip` with the run's stream count. Revisit forgetting's retention cost
   and long-run capacity at this scale; its learning measurements remain with
   the full training session. Measure admitted PartSpace rows
   on the target corpus and raise the current 32,768-row reserve before the
   million-sentence run. The required capacity depends on admitted parts,
   not one row per sentence; exhaustion now stops training. Record the chosen
   physical `nVectors`, occupancy and dictionary/optimizer memory cost.
   Target about 200,000 English word forms plus one associated object per
   word, with further room for non-verbal concepts: roughly one million
   physical ConceptualSpace rows (about 4.3 GB at width 1032 in float32,
   without Adam moments on the rotation-owned dictionary). Reserve a few
   hundred thousand PartSpace rows and measure WholeSpace admission for the
   same vocabulary and non-verbal coverage. The current CS 65,536 / PS 32,768
   settings are small-run values, not production capacity estimates.
   Set `SymbolSpace.ltmCapacity` for the intended million-row run; its default
   1,024 also sizes the shared `.when` ladder. Record retained-row memory, the
   actual subjective-clock range needed by the run and the ladder's resulting
   periods, so a long training history does not silently reuse temporal phases.
   *The content width* (never derived): exact
   clean-up of `k` bound items against `N` bank entries needs
   `D ≈ c·k·log N`; derive it for the run's vocabulary and run the R==0
   identity audit (of October 7) at that width on the run's config. Its
   reading decides the full training session's condition below
   ([FutureWork](FutureWork.md#expectation-by-inversion-and-the-reconstruction-audit-decided-2026-10-06-unscheduled)).
   Exit: the load and capacity measurement recorded, the width and the
   audit's reading, and
   `maxDocs`/`shardDir`/stream count/PS-CS-WS `nVectors`/`ltmCapacity` in the run config.
   *Added (Alec, 2026-10-08, from expectation across the architecture):* the age-appropriate corpus
   ([gradual training](FutureWork.md#gradual-training-an-age-ordered-corpus-read-in-stages-proposed-2026-09-29))
   is assembled here so the full training session reads it in order; the
   Viterbi toy
   shows the order, not only the content, is what lets one departure escape
   multi-step minima.


### 2026-11-08: formal arguments reuse general conceptual learning

Alec replaced the occupation-count/probability representation with explicit formal-argument concepts, initially zero, one native order above words. He clarified that upward member centroids and downward contextual influence belong to the general concept architecture; parts of speech have special construction and grammar use only. The implementation introduces `ConceptHierarchy` as that shared path, keeps form and meaning separate, reads operation trust from input argument radii, and leaves empty lift classes at zero gate readiness. Scalar uses remain telemetry. Schema 4 rejects count-only checkpoints for producer replay. Acceptance awaits Alec and Claude's review.

### 2026-11-19: one structure of Pi, Not and Sigma; the percept's code on two lanes; content-based looks

Item 1.7 replaced the first layers. Every order is a Pi, a Not and a Sigma: basic percepts are rows of the first Pi, nominated by MTM on recurrence, each bringing one concept; concepts are one signed value read by a dense Sigma at order zero, with signed weights and no bias; symbols are split from concepts and have no dimensions. A percept's code became 64 coordinates for parts and 64 for wholes on two lanes, evidence for and evidence against (Alec), replacing the 96 + 256 layout, the end-marked pairs and lengths, and the property code by bytes. Attention became a look over the sentence's percept bank: a request over concepts carried down as a salience map onto what and where, with the where on fixed wires from one parameter; the bracket table and its acts went. Alec asked that the documentation be brought into accord with it.

### 2026-11-24: the `.where` in one space of three dimensions

Until item 1.6 a place was one whole number on one line of named ranges, written as four wires, and each modality was to keep a perceptual space of its own with its own number of place axes ("the .where for imagery has to be 2D"; "leave each modality in its own perceptual space with varying .where, and integrate only at the level of zero-order concepts", Alec, 2026-11-13): a place on two axes was a row-major address decoded by a declared shape, and no vector was widened. On 2026-11-22 Alec decided one space for every model ("I would like to see .where as a 3D space for all models, it simplifies things"), and item 1.6 built it as its first round: a place is three whole numbers with four wires to an axis, the band of a percept is sixteen coordinates for eight, every declared width that embraces it is 8 greater, the fixed parameter `WhereRegistry.code` is `WhereRegistry.grid`, a range of the registry is a box, and a numeric image lies on two axes. Asked whether words should be moved out of the range of intensity or a dimension added for the modality, Alec chose boxes that do not overlap and no such dimension (2026-11-23). A model saved before this has the old widths and is not read; no migration was written. The design, the implementers' returns and the review are in the item 1.6 plan (`doc/plans/2026-11-18-item1-6-felt-math-world.md`, §16 and §17).

### 2026-11-24: a scene beside the words reaches the input

Alec decided that the percept of a pixel is its value from 0 to 1 and nothing of its place ("just a value, 0..1 like other percepts"), that an image holds its pixels as a sentence holds its words, and that the modality is intensities, of which pixels in an image are the first case. The first step built is the way in: a row of input may carry a `scene`, a model declares the scene's box with `architecture.sceneShape`, and the registry sets the box after every other range. Nothing read the scene at that step. The record is `doc/plans/2026-11-18-item1-6-felt-math-world.md`, §19 to §21.

### 2026-11-24: one table of values; an image's bank; the Pi over values

Until item 1.6 a number was a level percept keyed by its place: each place had a table of values of its own, on coordinates drawn for it, so that the same value at two places shared nothing, and the numeric stem kept a concept for each input place and set a value's two lanes by hand. A number was felt through a logarithmic chart. Alec decided otherwise: the percept is the value alone, in 0..1 and taken linearly ("the log chart was to avoid clipping, and sensation seems to be log. But we can treat that as a preprocessing step"); the values are a mereological space of their own with a Pi of their own; a place is told by its wires; and one Not follows. The numeric stem was rebuilt as an image's bank of occurrences read by that Pi. A fresh model then reads nothing of an image until its places have recurred. The record is `doc/plans/2026-11-18-item1-6-felt-math-world.md`, §19, §20, §22 and §25.

### 2026-11-27: a unit delay, one update of the field at each look, and what `attend` reads

Item 1.4 was to learn verbs as changes, and until this date nothing in the model held what had just been perceived: a field was read at one time, and a concept could be made of what was together but not of what followed what. Alec decided a unit delay on percepts, basic percepts and symbols alike ("temporal differencing"). Each basic percept one step back is a second input of the order-zero Sigma, on the numeric stem too, and each concept one step back is a column of the concept store, so that a kind may have for a part what was active a step ago, and is seated from what was active then. It is on by default (`architecture.perceptDelay`), is cleared by a hard reset, and is saved with a checkpoint. It is the one exception to the rule Alec set the same day, that all state is in STM, MTM or LTM: STM within a sentence, LTM between sentences.

The delay steps once at each look, and with it the conceptual field has one update at each look, the opening being look zero. Alec's order: the kept activation decays; the field rises through the structure, each order's new activation met by expectation (the nonlinear expectation already built, not a gain on a linear filter); it descends masked by `attend`'s own prediction, with no mask while attention is fully open; what is left is kept for the next look. Priming is the kept, decayed set (an AR(1)) and the delay is the observed field one step back (an MA(1)); the spread of priming is that traversal up and down, and no step of its own. What is kept is what the request of `attend` reads of the looks before, where it read a memory of its own.

The request of `attend` also reads the whole STM stack (eight entries), through a weight that starts at zero, so that the reduce after one word may inform the look at the next. This is the first part of what Alec asked of reading: "attend() should read one word at a time, and basically do a shift() in the compiler sense". It is not yet trained, and the loop of looks within a sentence is still there.

The links between kept ideas are written (`architecture.ideaTransitions`, off by default): a link from each kept idea to the next, in the one `LTM1`, marked so that it is read as neither a part nor a kind. They are relations (M1) over ideas (M0), "a graph of transitions which is not yet mined"; no count is kept until the chain is consolidated. Sequence is by this the order relation over symbols ([Orders](Orders.md), §13), with LTM for what is linear and concepts for what branches.

Also in this round, and in no narrative document yet: place visits are kept for each stream (`architecture.placeCapacity`, 1024), since two streams had shared one count; the whole-forward compile refuses more than one sentence to a stream; and `make test` writes the log that `make push` reads, a failure passing only with Alec's written ok in `test/failures_ok.txt`.

Decided on this date and not built: `attend` and `output` as operations of thought, with no reading loop apart from the thinking loop; intensity as an input and an output beside text; membership read off the weights of the symbols' Pi and of the Sigma, with no member list; verbs as some 24 classes of operation on their nouns; the meta-concept from mutual implication; behaviour from LTM by way of the basic level. These are in [Glossary](Glossary.md) (unit delay, look cycle), [Params](Params.md) and [Orders](Orders.md) where built, and not yet in [Architecture](Architecture.md), [Spaces](Spaces.md), [Language](Language.md) or [STM](STM.md). The record is `doc/plans/2026-11-23-item1-4-verbs-in-the-field.md`, §9 to §9.9 and §15, and the order of what is left is in `todo.md`.

## Work recorded in todo.md until 2026-11-26

*This is the content removed from `todo.md` on 2026-11-26, kept verbatim and in its original order: the paragraphs under "Countdown to the full training session", the blocks of the integrated items 2.5, 2.3, 2.0, 2, 1.8, 1.75, 1.7 and 1.6, and the blocks of the live items 1.5, 1.4, 1, 0.5 and 0 as they stood before they were trimmed. Only the link targets are changed, to be relative to `doc/`. The "Done" list stayed in `todo.md`.*

### The countdown preamble, as it stood in todo.md

Items count **down**: the next item to do has the highest number and **item 0
is the full training session**. (They are bullets, not a numbered list, because
Markdown renumbers a numbered list upward.) The order is the intended order; an item may
be taken early when it does not depend on a higher-numbered one. Codex builds;
Claude writes the specs and reviews each landing (Alec, 2026-09-21). "Done"
lines below keep the numbers their items had when they landed.

**Plan to the long run (Alec, 2026-11-09):** "The transformation-games note
and all other testing and harness work can move to 0.5 which becomes a test
development pass that can happen on Codex, so that you can do the compiler
work in 1.0 at the same time." Item 0.5 is new, between items 1 and 0
([spec](plans/2026-11-09-item0-5-test-development.md)): Codex builds it
while Claude builds item 1, an exception for this pair to the division of
2026-09-21. It takes the test and harness work that sat in items 2, 1 and 0,
the generated corpora (transformation games, the synthetic math problems
corpus) and the training corpus (the synthetic examples first, then the
corpus NanoChat trains on). Remaining order (Alec's answers of 2026-11-09,
spec §15): Codex finishes 2.0, then builds 0.5, with item 2's remainder in a
gap between its rounds; Claude builds 1.5 and then 1 in a second, detached
worktree; then 0, which trains three analyzable stages (words only; simple
sentences and synthetics; math), each read operator by operator and saved
with tests, before the million-plus sentence corpus.

**Tests until the long run (Alec, 2026-11-18):** "We need to run only
targetted test from now on during the point build rounds (e.g. 1.x), and
the complete sweep only at the item numbers (e.g. 1.0). No Run_Slow until
after all architecture is baked at the submission of 1.6 (and lets
schedule that for an overnight run)." Corrected the same day: "Sorry,
every .X should get a 40 minute sweep, including 1.8, but not before its
potential push round." So a build round runs the certificates of its
text, the tests of the files it changes, and both stopping gates; each
item, 1.8, 1.7, 1.6 and 1.5 among them, has one complete default sweep,
at the round that may be pushed and not before it; and the slow tier is
run once, overnight, when item 1.6 is submitted (1.8 specification §72;
[Testing](Testing.md)). For item 1.8 that round is the return of its
last text, §65.4.

**Current (2026-10-29):** Items 3.5, 3.0, 2.9 (BasicModel `17e9454763`)
and 2.8 (`a8a09a6acb`) are LANDED. Item 2.7 is closed code-complete at
review freeze `472f5000850e`; its carried work and November landing round remain below. Item 2.5
round A starts from the commit of that reviewed tree, including
D8's graded answer cost. **Update 2026-10-30:** round A LANDED at
`3718d9d0dc` (D9 masks `intuit` until round B);
round B and the one-backward round run as one Codex text under
the decisions of 2026-10-30: `intuit` a top-down Sigma over Pi with its own weights trained by
output, `ActivationPredictor`'s prediction the bottom-up spread replacing
location diffusion, D7 to future work, the cautious §16 package. No `RUN_SLOW` before
November.
**Update 2026-11-03:** item 2.5's completion round is in review. Item 2.0
below, from the 2026-11-03 decisions, follows it.
**Update 2026-11-03 (evening):** every item above 2.5 (9, 8, 5.5, 4, 3.5, 2.9,
2.8, 2.7) collapsed to a Done line after an audit of its remaining work; the
blocks as they stood are in
[DevelopmentHistory](DevelopmentHistory.md#work-recorded-in-todomd-until-2026-11-03),
and the remaining work is in items 2.0, 2, 1.5, 1 and 0, each marked
"Carried ... 2026-11-03". The §41 return is reviewed.
**Update 2026-11-04:** the §43 return is reviewed. Alec decided
§42's (a) to (d): the decomposition teacher retires in 2.5 with the chooser
at the fixed residual argmin; the uniform fraction stays .5; a whole is read
at its presence; and, with (c), one rule for evidence, "evidence belongs to
its source" ([Architecture](Architecture.md); the alternatives in
[FutureWork](FutureWork.md#belief-across-the-epoch-boundary-perseverance-and-testimony-alec-2026-11-04-revisit-if-necessary)).
The Codex text is §44.1: items 65 to 81, then the freeze.
**Update 2026-11-04 (evening):** the §45 return is reviewed. The repairs are
built, with a regression in the presence read's inverse, three gaps in the
evidence rule and a byte axis sized for a sentence rather than a word.
XOR_grammar fit 5 failed because its read-back decodes the last training root
against a dictionary that moved in the final update, where the walk policy was
never trained. The Codex text is §46.1: items 82 to 91, then §44.1's 71 to 73.
**Update 2026-11-04 (night):** the §47 return is reviewed. The repairs are
built. Fit 5's held names keep their first-sight meanings while the dictionary
moves. Alec decided that MTM holds references to STM and LTM and computes
identity from them: names now (item 96), the full form in item 2.0. A
headless composition's identity-0 alias was found in passing (item 94). The
Codex text is §48.1: items 92 to 97, then 71 to 73. For item 2.7: the word
surface capacity of 64 raises on 144 corpus percept runs (letter runs up to
126 bytes, high-byte runs up to 453); the corpus round constructs 512 or splits
such runs.
**Update 2026-11-05:** the §49 return is reviewed. Items 92 to 96 are
built. The default suite, which had not run since 3 November, fails 56 cases from
four rounds plus one memory regression; each is triaged with its repair. Alec
decided:
- `make test_all` leaves 2.5;
- six workers are allowed, recorded, if ten cross the memory ceiling;
- compile-guard raises are repaired at their producer;
- an identity refers to its known individual and to a kept thought's idea;
- a description reads its own noun phrase;
- MTM associates only on maximal overlap, now.

The Codex text is §50.1: items 98 to 108, then 71 to 73, closing 2.5. Claude
rebased `doc/DevelopmentHistory.md`'s 146 links (`4502b1c566`).

**Update 2026-11-05 (evening):** the §51 return is reviewed. Items 98 to 106 are
built. Claude's suite on `e0fc4a92719e` completes with 24 failures, each ported or
repaired in the next text. Item 107, a structural chain over a word not yet learned
read as that word's definition, closes 2.5 with the recorded single-writer contract
for definitions. Alec placed trust in a new item 2.3: a concept's trust is its
magnitude, and operators need their own. By the countdown, 2.3 comes after
2.5 and before 2.0. The Codex text is §52.1: items 109 to 115, then 71 and 73,
closing 2.5.

**Update 2026-11-05 (night):** the §53 return is reviewed. Items 109 to 115 are
built, and item 71 reports no event at or above 1e8. Claude's suite on the offered
freeze `8e9fe2634709` fails the forced pending-premise case in its `[True]` order: a
forward reference is read as its own mention. The defect is older than this round
and reproduces exactly from its captured stream. The freeze is not accepted. The
Codex text is §54.1: items 116 and 117, then the suite, item 71's runs and the
freeze.

**Update 2026-11-06:** the §55 return is reviewed. Item 116's repair
is approved; the forty pending-premise runs, the suite, the XOR fits and the three
float32 arms pass. The float64 discriminator, from the stream that completed on
`8e9fe2634709`, now diverges at sentence 17 and raises at sentence 52: a compose
bind names an introduction that no closing site writes. No freeze was offered. The
Codex text is §56.2 (Codex's diagnosis confirmed; it replaces §56.1): items 118 and
119, then the suite, item 71's runs and the
freeze.

**Update 2026-11-06 (morning):** the §57 return is reviewed. Item 118 is
approved, and the freeze `e56b69715947` is accepted: **item 2.5 is complete**.
It lands on Alec's word, as recorded below. Testing (Alec, 2026-11-06): until
the implementation is finished, each round runs only fast checks: direct
before-and-after cases, the targeted tests, and make test once at a freeze.
Long runs (item 71's), repeats and stability come later. The Codex text is §58.1.

**Update 2026-11-06 (afternoon):** **item 2.5 is LANDED** by Codex: BasicModel
`1e02cebf5d` and WikiOracle `b126029` (pointers: basicmodel `1e02cebf5d`,
nanochat reset to the fork's `7be4dc3`, openclaw `87b9a89`), pushed. Item 2.3
is decided and starts with
round 1.

**Update 2026-11-06 (night):** Codex's 2.3 round-1 return is reviewed.
The mechanisms are built as
specified and certified; the one `make test` fails 82 cases, traced to four
producers: relations and bookkeeping rows counted as views, trust applied to
charts without a meaning band (where it reads identically 0, which is why the
pending-premise pair fails), the operand's trust read from the derived vector
instead of the radius, and the pair sealed against all later evidence. Claude's
design decisions D1 to D5 (views as counters; trust in the meaning lanes only;
the radius is the trust; a non-concept operand contributes 1; no lexical
identity for whitespace or punctuation under either `whitespaceUnits` setting)
amend §6. **Q1 for Alec:** where operand trust acts; as built, the same sentence
cannot define after its word is learned and a configured truth is dead on
arrival; Claude recommends composing operand trust at the read, the written
pair kept. Alec answered the same night: the sealing flag goes; a
relation's pair is fixed relative to the relations it enters into, the trust of
"a wug is a bird" is $\min(x_1, x_2) \times x_3$ over uncoupled stored values, and
an order on the pairs maintains the relations; §9.3 is the design (stored and
effective pairs, the sort, the threshold at the index). The Codex text is §9.4.

**Update 2026-11-07:** Codex's completion (§9.4) is reviewed:
§9.3 is built and certified (28
certificates; the pending-premise pair passes once each on the final source); the one
sweep failure, the XOR convergence loss, is traced to the against lane carrying the
radius (a defect from §6.1's wording), which also makes `negate` the identity on a
stored concept; empty kinds cannot gain trust (a view of empty content counts 0).
D6 and D7 repair both; three weak ports are named. **Q2 for Alec:** whether a sign
whose name is not yet a denotation keeps its token pending instead of minting a new
identity (recommended). The Codex text is §11.1; the freeze follows it.

**Update 2026-11-07 (evening):** Codex's §11.1 return (§12) is reviewed:
D6 and D7 are built and certified
(33 trust certificates); the offered production source is unswept (the lookup changed
after the sweep). The four sweep failures resolve to: a seatless operand contributing
zero where D4 says one (the landed captured-verb test passes under the one-line rule);
D5 incomplete at admission (punctuation forms are still minted as words, so each is a
seat with no view and no grounding); and D8, operand trust on an untracked chart is one
(D2's norm reading scaled pairs on charts with no trust and made exact-one contracts
machine-dependent). `test_convergence` is unattributable (`MM_xor` is untracked,
unseeded, variable history); one intermittent is on watch. Not accepted; the Codex text
is §13.1; Q2 still open.

**Update 2026-11-07 (night):** Codex's §13.1 return (§14, round 1d) is reviewed:
the seatless rule, D5 at admission
and D8 are built and certified, the thirteen ports are sound, and the frozen sweep is
red by one native-grammar case. That case is a defect in the §9.3 build, not noise: the
index's threshold early-return withholds a named word's identity maps along with its
binding, `define` pops the pending entry, and a second naming rebuilds the description
without its form, so a word defined on a one-view object (effective $0.2835$ against
$0.5$) can no longer be found by its form and the next reading mints a duplicate
identity. The same state is behind two more intermittent native-grammar cases; all three
reproduce on the live source and the 1c candidate and never on the landed tree. D9: the
threshold gates the word $\to$ object binding only. Not accepted; the Codex text is
§15.1 (round 1e); Q2 still open.

**Update 2026-11-07 (night, later):** Alec answered Q2 as (b): a sign of a name below the
definition threshold stays a pending token and joins the name's object when the index
admits its named row (built in round 1e). He also fixed the
curriculum's operator order: nouns and verbs before the other parts of speech, that is
word-concepts, then object-concepts through `interpret`, then the verb operator, then
the modifiers, each operator introducing its parts of speech; recorded with
the lesson table and three opens for Alec, superseding
the earlier NP-first order. Later that night Alec resolved the three opens: what is
ordered is the operations learned, not the parts of speech; questions go in early
("bill ran." / "who ran?" tests comprehension, which teaches syntactic category); simple
before complex, enforced in part by the architecture (words before operations, NP + VP
before other operations) and in part by the corpus presentation (child development, the
Piagetian stages); intransitive before transitive. The lesson table is reordered accordingly
(`lift`, `ask`, `verb`, then the modifiers); round 2's architectural gates are bounded to
the coarse order (Claude's reading to confirm).

**Update 2026-11-07 (night, §16/§17):** Codex's round-1e return (§16) built D9 and
stopped the round on my item-2 wording ("record the producer and stop" meant the item).
Reviewed: D9 as built is approved
(80 ported and new cases pass here; the two native-grammar probes that failed now pass
10 of 10); finding 2's producer is the lexical span mask of `cs_read_memberships`, a
percept reader whose input set D9 restores, no publisher change; the three-document case
still fails 1 in 3 with a different producer, a witness reading kept at a word transaction
where interpret should be mandatory in reading mode, left to Codex to attribute. Not
accepted because unswept. Codex text §17.1 (round 1f): Q2(b), finding 3's attribution,
targeted run, one make test.

**Update 2026-11-07 (morning, §18/§19):** Codex's round-1f return (§18,
offered source `4abb6a6b…`) built Q2(b)
as §15.1 item 5 with four certificates through the sign close; reviewed:
approved as built (84 cases pass
here). Finding 3 is closed as attributed by my traced diagnostic: the untrained chooser
takes `sign` at the word's leaf, the Architecture's legal second decision, and the
three-document test asserts a free chooser outcome; its port (fixture fixes the symbol
decision, or `RUN_TRAINED`) and the sign logit's response to the radius are round 2's.
The one `make test` (nothing deselected) is 7451 passed, 1 failed: `test_doc_links` on
§18's links to receipt files written after the sweep, which passes on the live tree now;
rule from now on: every linked file exists before the sweep. Recommendation: accept the
freeze at `4abb6a6b…` as round 1's frozen source (alternative: one more sweep after the
receipt pre-creates its targets); Alec's word pending. Next: round 2's Codex text
(gates bounded to §6.3.2's coarse order) and the §5 review before the push (D8, D9,
Q2(b), the pending null-kind state).

**Update 2026-11-07 (round 1 accepted; round 2a, §20):** Alec accepted the freeze
("Great, round 2"): `4abb6a6b…` is item 2.3 round 1's frozen source, uncommitted.
Round 2 is two freezes: 2a builds
operation trust (decision 2; definiteness corrected to distributional, the
categorizer's probability of the argument over its occupants; `interpret`, `sign`,
`bind` and the fixed relations contribute 1; `trustViews` scales the use count; the
operation recorded on the row and entering the effective pair at the read),
`RUN_TRAINED` (`@pytest.mark.trained('<phase>')`, the §5b lessons as null phases, the
moved tests listed) and the curriculum measurements (reported, acted on by nothing);
2b builds the three gates after Alec's word on their data (fractions, closed-by-default,
the chooser question). Codex text §20.1; Codex's return is §21.

**Update 2026-11-07 (§21/§22):** Codex's round-2a return (§21,
source `7b4ccfa1…`) built operation trust,
`RUN_TRAINED` (eight tests moved, ten null phases) and the measurements; reviewed:
approved as built (96 cases pass here;
the minimum, not my 3(a) product, is the rule). The sweep is red by 49 in five attributed
groups: D10 (a row's learned operation is its first writer's; a later write merges and
leaves the tag), D11 (the naming write of `define` clears the tag), the thought-boundary
re-attachment guard (45 cases; keep the bank's catalogue and owner when none is passed),
one decision-5 fixture port (`test_ltm_consolidation` user ingestion) and one state-key
port. Finding 6: the three-document fixture holds against `sign` (5/5 traced) but the real
test raised once in three with a negative clause order from another free unary choice at
the leaf; Codex attributes the eligibility and narrows the fixture's menu to what reads
the sentence. Codex text §22.1 (round 2a-ii); return §23. 2b opens unchanged.

**Update 2026-11-08 (§23/§24):** Codex's round-2a-ii return (§23,
source `0218bfb5…`) built D10 (first
writer owns the tag), D11 (`LTM1.clear_operation` on both `define` paths), the inert
ownerless attachment, the two ports, and finding 6's repair: the chooser had been offered
the dummy field bind on an unknown-order mention; eligibility now checks the bind result's
order, and the three-document fixture's menu is STOP, interpret, union. Reviewed:
approved as built (200 cases pass here;
three-document 4/4). All 49 prior failures pass; the sweep is red by 2: one fixture
(`test_identity_threading` bind locations) deposits referents without an order, which
production never does; a port. Codex text §24.1 (round 2a-iii); return §25; a green sweep
freezes 2a. 2b waits on Alec's answers to §20's opens.

**Update 2026-11-08 (§25/§26, round 2a frozen):** Codex's round-2a-iii return (§25) ported the bind-location fixture
(order 1 on each antecedent) and swept green: 7470 passed, 323 skipped, 1 xpassed, exit 0,
nothing deselected. Accepted: round 2a
is frozen at `bfd6903f…`, uncommitted. Next: Alec's answers to §20's opens (gate data,
closed-by-default, the chooser), then round 2b's Codex text (the three gates), then the §5
review before the push (D8–D11, Q2(b), pending null-kind state, checkpoint schemas, bind
eligibility, attachment rule).

**Update 2026-11-08 (round 2b text, §27):** Alec asked to finish 2.3 without answering
§20's opens; the text carries the defaults as
stated assumptions (all data): two gates in `<gates>` (interpret: stable-word fraction 0.5,
caps the named rows; clause: mean definiteness 0.75 of `lift`'s classes, caps every row
whose operation is not in `except`, default interpret sign bind lift), closedTrust 0.1 in
the row's minimum, open stays open (checkpointed), **open by default in model.xml and
closed in the lesson configuration** (a reversal of §20's recommendation: closed by default
would gate every definition and the learning certificates for no mechanism gained), the
chooser unchanged, the tag now recorded for fixed relations too (trust 1 by name). Codex
text §27.1; return §28; then the §5 review before the push.

**Update 2026-11-08 (§28/§29, round 2b reviewed):** Codex's round 2b return (§28,
offered source fcc65c54..., 1002 files) built the
gates as §27 specified: `<gates>` in model.xml/xsd, `TrustGates`, `MM_corpus_lessons.xml`
(closed starts, the runner's default), LTM1 layout 5 with `gate_open` in the extra state,
the cap in `refresh_effective`'s minimum, fixed relations tagged, one port; one green
`make test` (7483/323/1 xpassed). Review §29: not accepted by one defect in my §27 text,
finding 7: the clause measurement inherits round 2a's empty-class definiteness of one and
opens the clause gate on a lesson model's first effective read, before lift is used; D12 makes
an unoccupied class measure zero for the gate (operation trust unchanged). Finding 8: an empty
`<except>` is rejected although Params allows it. Codex text §29.1 (round 2b-ii, return §30);
green there freezes round 2b and completes 2.3's code; then the §5 review before the push.
Observation for §5: one stable word of one opens the interpret gate (a `<minimum>` datum if wanted).

**Order and branches (Alec, 2026-11-03):** work happens on main, with no
sandbox branches; the countdown order stands, 2.5, 2.0, 2, then 1.5, 1 and 0;
when item 2.5 completes, everything is on main, committed and pushed, then
each sub-repository is synced and the WikiOracle pointers are bumped and
pushed. `nanochat` carries no changes of ours (Alec, 2026-11-03): its fork
is upstream of 2026-02-19 plus Alec's two commits (the subpath `API_URL` fix
and the training-artifacts ignore), the recorded pointer `7be4dc3`; the local
checkout carries upstream to 2026-07-03 with the ignore re-applied and lacks
the `API_URL` fix, so its sync is a reset of the local checkout to the
fork's commit `7be4dc3` (Alec, 2026-11-03), with nothing pushed and the
pointer unchanged.

**History (2026-10-24):** Item 3.5 is LANDED. Iteration 1 published at BasicModel
`f306fc28aa`, WikiOracle `319a793`; iteration 2 landed 2026-10-24 at
BasicModel `75345f891e` (published main
`4f136f76faabafc24397765f4ffe90d01d3dc4bc`, WikiOracle
`69f82f5f77a3980020841fe859a09afb7733e7be`) after Claude's LAND review:
the second repair's five scoped groups pass **588 cases with 41 skips and
zero failures**; the one end-of-3.5 sweep (`0bd65c5e2c`, 35 failures) is
repaired, nothing retired.
History note (2026-10-25): history after `f306fc28aa` was rewritten before the push (`537523ae04`); `75345f891e`, `a34325c85d` and `f095569a1f` are respectively `2067dbc25a`, `3561f39aa0` and `31c3e4a59e` on origin.
Learning remainders (line 3's description 1/4, the modal corpus 0/4, the
six placements 0/6) are item 0's with their budgets; layout/removal, STM
chunks, overlay retirement and the certainty record are item 3.0's.
**Item 3.0 steps 1–3 are committed (Alec’s hand-off); step 1 is approved** (Claude's review, 2026-10-25):
layout/removal source SHA256 `34848e154f0fa9912223d4117136d7eb63657cefaeca94a76c6574559a73819e`,
**911 passed, seven existing prerequisite skips, zero failures**.
**Step 2, placement, is approved and committed**
(Claude's review):
source SHA256 `1e94238b57c99ba2b1003de2467e6f6acf543b68dc425263bed708509677d55e`,
**1,003 passes, seven standing prerequisite skips under the documented protocol**.
Step 2 replaces
the interim context update. **Step 3's repair is approved**
(Claude's review),
source `4b7fa2aaf7`: 1,113 passes, seven standing prerequisite skips,
32/32 complete premise readings. Its two closing items:
placement's greedy coverage is also 1.0, so the **6/8 companion flag is
item 0's learning work**; estimates now carry zero time without consuming
an arrival tick, while current time remains a predictor input.
Closing source `cd49a1e84534d6c92f074276e2cfc57fccd1226a876bad4dce5905c6d6b3107d`
passed **1,113 cases with seven standing prerequisite skips and zero test failures**, plus one smoke.
Step 4 source `44ea10180c` was rejected in §17: silence
entered the sentence keep comparison. Alec's decided repair excludes silence
from A while retaining its output training. Repair source
`acac0b3448f5a265f62a64bab6d0368563df0ab1ecb15d76d4990428ddc9d1f1`
finishes **1,490 passed, one failed, eight standing skips**, plus the smoke.
Step 4 is approved as a mechanism round.
The approved two-stream port closes gate 6: all premise readings are nonempty
and both first premises are full; its file and both decomposition arms pass
6/6, with companions 6/8.
**Step 5, orders learned**, is approved on final
source `c4a1d92e3fc27decc669322a507b362dd615556e89041331fe1259116584ab95`:
1,647 passed, ten existing prerequisite skips, zero failures, plus the native
smoke; companions remain 6/8. The failed first freeze is retained separately.
**Item 3.0 LANDED — 2026-10-26**, approved in Claude's review. The landing names step 5 `c4a1d92e3f`, serializer `caece43fc0`, sweep repair `03fbbf2a7d9c`, final model repair `6fe273d2f12a7`, and test budgets `ee7ca4e920bc`. The learning records stay at item 0 at their listed budgets; the four runtime failure groups move to item 1 as named below; the unforced 1-to-2 raise certificate for “Felix is a cat. Ciao is a cat.” moves to item 3's corpus round. The complete default sweep retains 6,386 passes, 324 skips and one permitted XPASS; final model repair retains its smoke, 1,812 scoped passes and ten standing skips. The ten budget-policy checks and reduced stability replay pass. Companions remain 6/8; no complete passing slow sweep is claimed. No new test round accompanies landing.

**Historical step 1b review ledger (superseded by Current).**

Step 1b's native identity candidate `027ce224d` and STM1 candidate
`dde893527` are on
`codex/item3-5-first-pass`. **The migration decision is
resolved:** Alec retired stage3, np_definite and np_description. They are
unregistered and archived beside their producing receipts, never to be
read; their certificates are prerequisite-skipped until their producing
lessons are next needed with fixed budgets/seeds. Rejection of retired
coordinates is permanent. The
STM1 follow-up records
the retirement and removal of the blanket cycle restriction.
**Step 1b landing remains:** the §6.10 close/read repair and saved
reverse-order replays pass. Both decomposition lengths pass in both arms
in attempt 124,
all eight companion questions correct. The one full sweep completed 5,965 cases with 219 failures: 218 are
passing scoped cases or declared ports, and the capacity repair is now
recorded in review 3.
The native-119 replay passes with the original eight seats and budget.
Saved-state paired-reconstruction evidence remains separate. The
pool-capacity question
is decided (ordinary referent stays an unminted LTM1
occurrence; mint only relation arguments at the close or by recurrence;
unseatable mint fails loudly; no capacity masking). Claude's third review
adds one blocking finding: the pending-premise certificate
(§14 e, both orders) fails at fresh initializations. Exact captures now
reproduce it; named-head, lexical-key and composed-marker repairs and the
working-read certificate port have passing regressions. The
live-exploration contract question
has the D1 probe above; D2 is decided but queued, and the
six-fresh-per-order gate remains open. Complete A without weakening
the assertion or selecting passing seeds, then the
§10.2 round
(Alec, 2026-10-20: the determiner's `lower` prepares the token to be
bound by identity in STM; no concept row and no edge for the token, the
lowering is inline; fixture back to 8 rows; the capacity certificate at the original inventory), then publish both
mains with the October 20 docs and a clean BasicModel tree.
*Fourth review (Claude, 2026-10-20 night):* B accepted as
scoped (`mm-05` at eight seats; the 8192 fixture withdrawn); A's three
mechanism repairs and the line-422 working-read port accepted; A stays
open on the keep probe for the forward line-391 case (step-0 precedent:
repair the judge, not the certificate), Alec's scope decision for the
reverse line-398 assertion (recommended: no row keeps a forward
reference naming x), then six fresh initializations per order.
*Fifth review (Claude):* the D1 probe is accepted; the keep is
decided by one byte of reconstruction with A=0 by design, so it is the
contract case; recommended to Alec: D2 now, the §10.2 round next (both
failing readings are the complement mint that §10.2 change 1 removes),
the twelve fresh runs on that source, and a contract split (mechanism
assertion in step 1b, the strict reading kept verbatim at item 0) only
if failures remain. **Alec decided: §10.2 first, contract later.**
*Sixth review (Claude, 2026-10-21):* D2, the eight §10.2
changes and the carrier follow-up accepted; D1 dissolved (both replays
tie and keep greedy; twelve fresh pass with the unchanged assertion).
The 43 scoped failures are representation ports (authorized under the
kept-semantics rule) plus one mechanism gap: a bound use is not grouped
with its antecedent; recommended: the mention's row keeps the selected
occurrence in `role_refs[0]` (object permanence by reference) and the
read follows it. **Alec decided (2026-10-21): ports authorized; the
grouping lives in MTM, a medium-term identity pool on ConceptualSpace,
independent of LTM**; then final evidence by hash on the
frozen source. *Seventh review (Claude):* MTM accepted as
§10.3 with both merge certificates passing unchanged; one repair before
the freeze (a re-closed occurrence must detach from its previous token)
and the fourth occurrence-view port authorized; then the final gates by
hash and publication. *Eighth review (Claude):* the frozen
source `eac2fe87b` is accepted (finding A repaired, port B applied, MTM
not implicated); reverse fresh draws 2 and 5 fail by the R-only keep
(A=0 on unlabelled premises, the ambiguity rule leaves x open under a
kept conflicting equality); recommended to Alec: force the keep in the
forced certificate and keep the live-keep form verbatim at item 0.
**Alec decided instead: `bind` is the only binder at the
arrival too; the reader's equality rule and ambiguity clause go; a
named site with no candidate opens an anonymous MTM token that the
arrival's re-run bind refines; forced like any operation in the
certificate; assertions, keep law and budget untouched.** Completed scoped work: transient full-width STM1 and close
dissolution; shared six-column Part store; cube-coordinate maps and
inverse checks; Part-derived exact bootstrap and saved/random paired
reconstruction; historical saved-state §14 checks; predictor retirement;
G1–G3 and reference-close. These are not a completed landing.
Alec's §6.6 answers withdraw the exact-leaf inversion and width gates;
identity computed in STM is never written back into LTM. All-LTM2
integration is implemented in this step, including the expression reader. The per-dimension withheld mark belongs in concept content
beside the lanes at the row conversion.
Production keeps the prime ON and uncapped,
P/S/R/D=0 and φ=.5.

*Ninth review (Claude):* the §10.4 core and its follow-up
repairs accepted; three repairs before the next freeze: bind is priced
by the failure to understand like every chooser operation (Alec: the
closing reference credit leaves A); refinement merges the pending token
into the chosen antecedent's token; the bootstrap zero-R failure is a
regression of the carrier hunk in `67c5b203f` (replay-bisected), to be
repaired. Alec, later: every noun adds an object to MTM automatically;
binding links it to an existing object or leaves it its own (a zero-logit
dummy), learned like any choice; the pending-premise certificate's
closure forcing is removed (no wrong binding is asserted; the strict
closure assertions move verbatim to item 0). Then every gate on the new hash.
*Tenth review (Claude):* frozen `575cab06c` verified (carrier,
regressions with the six authorized ports pending, bootstrap). Alec: a
joined site is never re-bound; incompatibility makes the mention a new
identity; re-identification is FutureWork; thought binds follow decision
A; "reconstruction reads the bind" is the recommended next round after
step 1b. Repairs: nested-role answer view, kind adoption on merge.
*Eleventh review:* §20 implemented and accepted; the series
stopped on a wrong binding (x to the space: markers were bind candidates).
Alec: bind is offered only on words the final parse identifies as nouns,
and only nouns are candidates (no ID → NP rule; identity stays an option);
the seven retained checks get the authorized dispositions (two thought
ports, one count port, four partition assertions to a re-identification
certificate under FutureWork). Then a new freeze and every gate.

**Naming sweep at the step 1b landing (Alec, 2026-10-21: "rename when
you think best"):** LTM1 → LTM0 (the occurrence store: order 0, no
identity), LTM2 → LTM1 (relations), the stale LTM3 retired; LTM1 is a
stratum of the LTM0 store (Alec), LTM2 = relations over relations;
renamed in place in the live contract docs, a naming line in the other
docs' amendment block, code comments by Codex in the same landing.

**Necessary deferral, accepted after evaluation:** the copular-identity
certificate needs its producing lesson. Item 3.5 owns stage 4e → NP
continuation → copular identity → artifact registration and unchanged c
certificate. No producing artifact exists; the October 18 decision withdrew
the proposed one-equality-hop reader patch in favor of this phase gate.
Graded versus shuffled (§9.6), derivation-optimality, long-run collapse and
the original failed native learning run remain at Item 0. Independent 5.5
learning gates remain separate. No extra training or regrade closes them. The
optional cold-start boundary learning certificate is also incomplete:
native-117 hit its 901-second
runner bound during the original six epochs. Carry it to Item 0's learning
evidence with that budget unchanged; it is not a step-1b mechanism pass.

**Active mechanism:** Item **3.0**. Step 1 is approved;
step 2, placement, is approved and committed;
step 3's repair is approved.
The closing replay moves the 6/8 companion residual to item 0; the decided
estimate now has zero time and consumes no tick. Closing source
`cd49a1e845` passes 1,113 cases with seven standing skips; both regressions, the unported clock test and the percept certificate pass.
Step 4 is approved in §20, and its approved two-stream port closes gate 6
with 6/6 scoped passes and companions 6/8.
**Step 5, orders learned**, is approved on `c4a1d92e3f`:
1,647 passed, ten existing skips, zero failures, plus the smoke; companions 6/8.
**Item 3.0 LANDED — 2026-10-26**, approved in Claude's review. The landing names step 5 `c4a1d92e3f`, serializer `caece43fc0`, sweep repair `03fbbf2a7d9c`, final model repair `6fe273d2f12a7`, and test budgets `ee7ca4e920bc`. The learning records stay at item 0 at their listed budgets; the four runtime failure groups move to item 1 as named below; the unforced 1-to-2 raise certificate for “Felix is a cat. Ciao is a cat.” moves to item 3's corpus round.
3.5 landed 2026-10-24 at BasicModel `75345f891e`. *3.5's record:* it ran on main after
Item 4 publication, in Alec's first-pass
order: bind as thought; LTM1/LTM3 and the row amendment; stage 4e and NP
continuation before identity; verb group element; tense/aspect and second
difference; then the remaining 5.5 mechanisms.
The bind step-1 landing
has completed Claude's review, one full sweep (5,907/5,907 completed;
48 failures repaired with scoped checks), and final native validation:
72 passed, one copular prerequisite skip; both decomposition lengths pass
in both gain arms with all eight questions correct. Publication to both
mains preceded step 1b's G0 and identity migration. On the earlier
identity candidate, G1–G3, the 18 reference cases and both decomposition
lengths in both arms passed on the native scorer; that candidate's full
sweep completed all 5,926 cases with 15 failures,
followed by scoped repairs of 11 failed cases. The three artifact failures
remain in that historical result; their artifacts are now retired, not
regraded as successful loads. The later STM1/Part round repairs the native
cube, final §14 read failures and expression-type defects. Its scoped
gates pass; the one landing sweep is complete and its capacity failure
remains open before publication. The physical LTM1/LTM3 split, context wholes, spatial extent and
situated occurrence lowering are carried into step 2. Steps 2–6 have not started.
There is no identity-learning claim: the native reference-close lesson remains
0/146 bindings, 0/20 row pairs and zero A columns.
*Status (Claude, 2026-10-20):* step 0 round 2 is accepted and published on
both mains (`bfc834d79`/`d6eb0cf3b`, WikiOracle `cb4d718`); step 1, `bind`
as a thought operation, is reviewed on `codex/item3-5-first-pass` at
`f4f11aa5b` — conforms to its design, gates G1–G3 and both
decomposition certificates pass; Alec decided it lands after its sweep and
**step 1b** migrates the scorer
(identity only in the major structures: the meet as the content term,
containment for the form term, no nouns dictionary, no count tables, no
ICA event chart, STM as an index) before LTM1/LTM3.
*Review of the step 1b tree (Claude, 2026-10-20, later):* step 1 is
published (`204e1be55` / WikiOracle `a89c958`, with the answer-kind judge);
step 1b is an unpublished candidate carrying the identity migration, the
retired artifacts, STM1 in progress, the `PartStore` (LTM2 and the
taxonomy as one owner), the cube chart, and model-time `.when`. Every
decision of 2026-10-19/20 was recorded
with its status and the commit it lands in; Codex's §6.7 read question is
answered
(symbol types and ordered pairs suffice; no metadata on Part). Open on the
branch: the live-cube counterexample, the §14 initialization read, the
expression read and the bootstrap read, the paired-reconstruction `.1`,
reverse-order premise resolution, `IntraSentenceLayer`.

The word-level evaluator built in 6.8 waits for the trained checkpoint;
acceptance does not authorize fresh-model bulk scoring. Every item above 2.5
collapsed to a Done line on 2026-11-03 after an audit of its remaining work;
what remained of them is in items 2.0, 2, 1.5, 1 and 0 below, with the
empirical gates of items 9 and 8 at item 0.

Current contracts, limits and receipts are in
[SelectedMeaning](SelectedMeaning.md),
[KernelRetirement](DevelopmentHistory.md#frame-kernel-test-migration),
[ExpectationRetention](ExpectationRetention.md) and
[Testing](Testing.md).

**Publish rule (Alec, October 11):** repair → one regression per defect →
scoped measurement → candidate commit on a branch → **Claude's code review**
→ resolve review → one full sweep at landing → BasicModel push → WikiOracle
bump/push, with the co-author trailer. Candidate commits are authorized before
review; their commit hashes are the archive. No tarballs, manifests or
integrity checks. Receipts contain gate numbers and one table per question;
diagnostics stay in files.
Natural word →
operator associations use the existing compose/generate grammars. Mechanism
probes do not satisfy learning gates, and a null result is recorded as null.
**No test pins a random seed in order to pass** (Alec, 2026-09-21): a fixed
seed may make a *measurement* reproducible, never an *assertion* true. A test
that fails at some seed has found a defect; fix the defect, or let it fail.
Do not remove unused reasoning methods without Alec's review.

**The grammatical operations and their inverses are still in development
(Alec, 2026-09-26).** The compose, thought and generate catalogs are a
fragment, their routing is learned with sparse credit, and their tied
inverses are exercised mostly by reconstruction. No item's gate, and no
conference number, should be read as a claim about the performance of the
grammar as a set yet; a poor result on wording, parse depth or free
generation is expected at this stage and is recorded, not tuned away.

### Integrated items 2.5 to 1.6, as they stood in todo.md

- **2.5. Thinking: derivation over chunks with trust, and `intuit`.**
   *Spec written (Claude, 2026-10-27),
   from Alec's eight answers
   and the earlier `intuit` design, whose
   leaf-index scan and `Mind.SERIAL` dependency it amends. D1–D4 decided;
   the 2026-10-28 readings on training, fixed LTM, order zero
   and the pick followed. Two rounds: A (the serial path,
   starting from item 2.7's code-complete commit at
   review freeze `472f5000850e`) and B (`intuit`, which
   at round B also replaces item 2.9's location diffusion as the
   spreading mechanism, consolidating conceptual expectation and priming
   into one operator; Alec 2026-10-28). Round A carries D8's
   graded binding cost plus text cost; 2.7's landing
   round remains in November. Decided direction (Alec, 2026-10-29):
   one error signal and one backward: stop-gradient for attention and
   expectation, jitter of reconstruction's strength against output, and the
   chooser's log-probabilities in the same objective. Round A landed
   2026-10-29 at `3718d9d0dc` (`chunk_truth`, `.where`
   addressing, `infer`, D8, D9). Decided 2026-10-30: `intuit` is
   top-down with its own Sigma-over-Pi weights trained only by output; the
   bottom-up spread is `ActivationPredictor`'s prediction; no open-pass
   `intuit`; D7 to future work; round B and the one-backward round in one
   Codex text.*
   The serial slot becomes the episode's focus chunk among the shared
   eight; `descend`/`return` follow tagged LTM1 references; operators
   `infer`, `refute`, `substitute` beside the catalogue; `intuit` proposes
   into chunks, the serial path confirms; trust computed along the path
   (minimum, maximum, lane exchange), learned only at the leaves; weak and
   strong forcing; the implicit route, pervasion over an object's parts and
   wholes in order-0 space, judged together with the explicit chain.
   Rounds sized at spec time. Exit: its gates green and
   one thought episode's records replaying a derivation with its pair.
- **2.3. Trust in operations** (Alec, 2026-11-05). A concept's trust is its
   magnitude. Operators need a trust value of their own, unless they exist as
   vectors, and a relation's trust composes its operands' and its operator's.
   **Decided 2026-11-06 (provisional until the review before the push):** a concept's lane length
   is set from its views and its stability (agreement among its occurrences'
   parts and wholes), training moving direction only; an operation's trust
   is the definiteness of its formal arguments, else its count; gates open in
   psychologically motivated groups, not one at a time (for Alec's
   review); trust composes by minimum and acts as a weight with one threshold
   at the definition writer; source trust stays separate; `RUN_TRAINED` takes
   the tests that assert learned behaviour, no lesson run; the serial
   reader's whitespace admission is repaired here. Two rounds.
- **2.0. One gradient, one choice mechanism, memory as the state of
   affairs, behavioral chains and basic percepts** (Alec, 2026-11-03 and
   2026-11-04). After item 2.5's
   completion round, before item 2, and before item 0: the gradient
   discussions Alec asked to settle "before step 0 large training", with
   the chains he placed "before step 0". Decided 2026-11-03:
   `separateChooser` false by default, one update per sentence; output's
   gradient projected one-sidedly against reconstruction's on every shared
   unit (concept rows, definition edges, operator tensors), in place of
   jitter, which stays at 0 as an optional annealing; generate's proposal
   and credit made the same as compose's, and `walk_policy` retired; the
   decomposition teacher retired, its chooser at the residual argmin; the
   verb and modal teachers retired, verbs and modals trained by their
   participation, credit crossing sentences through MTM by the re-read,
   the verb-group assigner kept; every choice reward whole on the output
   side; the answer walk's residual rebooked to output; the derivation
   inverse given the forward's action context; the five terms evaluated on
   the model's internals kept, recorded in
   [GradientFlow](GradientFlow.md#terms-on-internals). The order-zero
   loop ("thinking is optional"): reconstruction from order-0
   concepts, understanding a thought process from order 0 to order 0,
   output the same decoder on the result; compose, think and generate are
   System 2 on System 1; MTM is the state of affairs, for thought and
   action only; `intuit()` within a step and expectation across steps are
   the order-0 operations. Reconstruction then ceases to involve the
   grammar; its build, reconstruction from order 0 with two acceptance
   checks (an input's own order 0 decoded as thought's output equals its
   reconstruction exactly; reconstruction is unchanged with STM and MTM
   ablated), is the item's last part:
   reconstruction is input (thinking)* output, the input reconstruction
   with no thinking step and the order-zero reconstruction after
   understanding, in which symbols take part. **Behavioral chains in MTM
   at order zero**:
   STM, MTM and LTM in one file, `bin/Memory.py`; STM a buffer of eight
   ideas and relations; MTM, over its time horizon, knits STM's and LTM's
   entries into per-order identities that maintain a state of affairs, by
   one link function with no branch on order, each collection meant as a
   connected path (a time window and likeness enforced, place and
   continuity scored): an extended event when it joins states, an extended
   noun when it joins mentions, the precursor of concepts and taxonomic
   relations; promotion writes the path (membership and successor rows); a
   recurring chain promotes to a chained event concept; a thing that
   persists through a chain is a noun candidate. **Basic percepts** (Alec, 2026-11-04): the last
   perceptual layer, a preallocated table sized in `data/model.xml`,
   whose percepts tie parts and wholes at their centroid, taking over that
   role from the first symbolic layer; order-0 concepts collect basic
   percepts as higher concepts collect symbols, order-0 symbols represent
   them, and symbolic space has no mereological read; the order-0 reverse
   runs through basic percepts to bytes; one WholeSpace replaces the
   per-stage copies left from the 2026-04-22 refactor. Metrics for the round: held-out
   cost, acquisition, answers and each chooser's departure win rate, no
   chooser variance. Gates: the projection measured per unit (cosine before
   projection, fraction of steps projected, fraction of output's gradient
   removed), the realized relative step per level (codes, verbs, modals),
   the tests these changes touch ported with before/after, the gates of
   the loop, memory and basic-percepts rounds, `make test`. Prerequisite
   (2026-11-04): the gradient spikes item 2.5's arms showed with
   `separateChooser` false, item 2.0's default,
   attributed in item 2.5 and repaired before item 2.0's measurements.
   One Codex text, written after the review of item 2.5's completion round.
   **Carried into item 2.0 on 2026-11-03 (audit of items 2.7, 8 and 3.5):**
   the supplied `yes` admission defect (31 of 32 lesson-13
   absences are "yes" missing from the terminal bank): a three-stage
   report (admitted, shortlisted, exact route available) for every supplied
   form, with `yes` present in all applicable banks, and a certificate on
   the admission path (`Corpus.admit_training_forms` to
   `Generative.lexical_candidates`); home confirmed as 2.0 round 2 (Alec, 2026-11-08,
   "Yes, round 2").
   If the sentence-82 MTM rejection (`MediumTermMemory._value`,
   never captured, not recurred in eight later runs)
   recurs in the round's lesson-13 arms, capture the working value and pool
   state at `ReferenceClose.commit`'s deferred `mtm.bind` and repair at the
   producer; no clamp or fallback. The `XOR_grammar.xml` reconstruction gate
   (`test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`,
   four of four pairs in source order at grammar input and inverse, unseeded; open since item 2.5's completion
   round, recommended for repair by the decomposition teacher's retirement in
   item 2.5) is run before and after that retirement with the chooser at the
   residual argmin; its class-accuracy sibling passes; no seed or budget
   change. Output typing (carried from 3.5 line 7; decided, Alec, 2026-11-03):
   output types are per formal argument (`PartsOfSpeech.formal_arguments`,
   the `argout` faces beside the `argin` faces) and collapse to parts of
   speech as already described ([Language](Language.md), "Distributional
   parts of speech"); the `argout` prototypes are kept per formal argument
   and read by the same occupation measure as the inputs.
   **Rounds (Alec, 2026-11-05: "after 2.5, 2.3 then 2.0, nothing else";
   "Just divide the work").** 2.0 stays one item, built in rounds, each with
   its own freeze; the loop was the last, as recorded, until Alec added round 7 on 2026-11-10. Already done in 2.5,
   struck from 2.0: the decomposition teacher's retirement, the
   `XOR_grammar.xml` fits around it (2.0 still runs the
   gate once after `walk_policy`'s retirement), and the spike attribution.
   Held-out cost waits for item 0's start; the five terms
   on internals stay recorded. The rounds, in order:
   1. *Behaviour-neutral moves:* STM, MTM, LTM0 and LTM1 into `bin/Memory.py`
      (whether `ShortTermMemory` moves too is decided in the text); thought
      ideas stamped and found by `.when`; `ReferenceClose.commit`'s
      word-occurrence substitution and its `word_occurrences` argument
      retired, which no closing reaches since item 2.5.
   2. *Repairs and the measured "before":* any event at or above 1e8 under
      `separateChooser` false that item 2.5's lesson-13 and XOR runs
      report, repaired at its
      producer; fail loud in the reverse and `runBatch`; the derivation
      inverse given the forward's action context; one WholeSpace; the
      supplied `yes` admission; the metrics above recorded as
      the baseline. The sentence-82 rejection is a capture rule on every
      lesson-13 run. (Alec, 2026-11-08: "Fast checks only"; the measured
      "before" and every other measurement of 2.0 move to the end of the
      item, taken on round 2's frozen source and on the finished tree; the
      round builds the repairs with fast checks.)
   3. *One gradient:* `separateChooser` false with the one-sided projection
      and the sides. It starts from what item 2.5 measured: under
      the shared backward the output gradient reaches the percept-to-word
      readout and the answer readers, and none of the codes, operators and
      definitions; the projection leaves a momentum-driven tail (Codex's
      proposal: project outward momentum at clipped coordinates).
   4. *One choice mechanism:* generate under compose's proposal and credit,
      `walk_policy` retired and the XOR gate run; the verb and modal teachers
      retired (participation, $\kappa$ in the model file); output typing.
   5. *Memory and basic percepts, one round* (Alec, 2026-11-05: "In 2.0, 5
      and 6 merged, but MTM only operative for basic level percepts (not
      parts and wholes)"). MTM's one link function, extended events and
      nouns, promotion of paths and chains, the full reference form (with an
      elided role's content: under containment an elided role binds nothing),
      credit across sentences by the re-read, verbs over the state
      of affairs.
      Runs form concepts at every level (Alec, 2026-11-05): an event run's
      state is computed from its members as an identity's value is from its
      references, one order lower; events as mereological neighbours in
      world-space stay future work. MTM's link also runs over basic-level percepts, never
      over parts or wholes alone, and a recurring one promotes to a basic
      percept, which word admission creates today. The compose-time whole
      read and the per-word typed slots keep their logic and read basic
      percepts ("Why would this logic change?").
   6. *The order-zero loop*, input reconstruction then output through the
      same reverse.
   7. *Episodic memory* (Alec, 2026-11-10: "lets schedule episodic memory as a
      round inside 2.0"): a few percepts per sentence, their number a model
      parameter, read from the open attention pass (`FamiliarTier.open_sentence`),
      stamped with model time and read back by the time-only `query`, whose
      review makes them the attended field.
   **2026-11-09 repair return:** the pinned-bank repair and twelve ports pass
   their targeted checks, but the one sweep stopped at the aggregate memory
   guard (28.29/28 GiB), 6,955/7,941 cases completed, 28 failed. Round 6 and
   the end-of-item measurements were not started; no retry, commit or push.
   **2026-11-09 second repair:** the 256-word bound stays; the runner
   now uses six workers with 12 GiB per worker and 28 GiB aggregate. Fixture
   ports, the fixed compiled bank, and trial/commit order routing were swept
   once: **5,615/7,943 completed; 5,391 passed, 218 skipped, 6 failed** before
   the aggregate guard stopped it at **28.056/28 GiB**. The six failures have
   three diagnosed causes (span stream selection, local introduction order,
   and the forced empty-search fixture). Twelve fresh-worker comparisons
   and constructor allocation probes attribute the memory increase to the
   existing round-5 stores with the inherited 256 bound; the seven-model
   audit has identical constructor storage before and after this repair.
   Source `a48eff54…` remains frozen and unaccepted. Round 6 and the
   measurements were not started; no second sweep, commit or push.
   **2026-11-10 third repair:** the approved three-worker default
   completed **7,955/7,955 cases** with **zero memory stops**: **7,627 passed,
   325 skipped, 1 xpassed, 2 failed**. All required gates passed. The archive
   origin fixture was ported and its full file passed **1,003/1,003**;
   the item-1.5 layout failure remains carried. Five measured large cases
   passed in fresh workers. The 256-word bound, 28 GiB aggregate and 8 GiB
   reserve remain unchanged. Round-5 return freeze **`0cef3933bef3…`**;
   twenty XOR_exact repetitions at its immutable archive returned **19 passed,
   1 failed**. Round 6 now implements the common order-zero decoder, separate
   generate writer, two field losses, post-thought expectation and timed STM
   thought ideas. Its full sweep completed 7,972 cases; 101 of its 102 failures
   now pass in affected-file reruns, with item 1.5's feature-layout case carried.
   The two-source measurements exposed stored-reference, convex-mean, thought-link
   and surface-only introduction defects; their repairs, captures, direct checks
   and affected-file reruns are retained. Final source **`afb18256c7ac…`** is
   frozen for review. All four final lesson arms completed **160/160**, with
   **0 recorded ≥1e8 events**, but **0/320 correct answers** and no acquisition
   criterion met. The baseline lesson arms stopped at **8, 16, 2 and 16**
   sentences, so no full-length learning comparison is available. Eight native
   400-epoch XOR fits gave **2/8 → 6/8** numeric passes, all **4/4** word
   multisets; every draw is retained. The final twenty XOR_exact repetitions
   passed **20/20**. Claude's concurrent diagnosis identifies three
   further post-sweep regressions, including a grammar gate that now reads the
   field decoder. Those findings and the grammar-inverse gate remain unresolved;
   the source stays frozen for a separate follow-up text. The return
   makes no population-level learning claim. No commit or push.
   **2026-11-10 follow-up (replacing the unapplied follow-up text):** restore the
   grammar inverse → generated occurrences → native reverse for order-zero
   reconstruction and the grammar gate, beside the unchanged System-1 field
   reconstruction. Restore reconstruction walk residual and credit on R; keep
   the output bookings on A. The attribution-share and positive-definition
   centroid contracts follow Alec's decisions. The reported failure groups
   and dead synthesis methods are repaired. The unchanged grammar
   gate passes **4/4** on its captured failed draw; the field audit is separate.
   Two measurement-exposed candidate/commit mismatches have producer repairs,
   whole-file checks and passing original-stream replays (**38/38**, **104/104**).
   A concluded-thought definition now finishes before epoch withdrawal;
   its original failed stream passes **33/33**. A later consolidation-order
   failure has a producer repair, **82/82** targeted passes and a **78/78**
   original-stream replay. The next panel exposes prepared-arrival order and
   event-token reads; both have producer repairs, **25/25** direct passes and
   **135 passed / 4 skipped** in their affected files. The event stream
   passes **92/92**; the arrival replay exposes a duplicate planned-member
   offer, now repaired with **162 passed / 4 skipped** and an **85/85**
   original-stream replay before the final measurement panel.
   The next event-link failure exposes a word-trust update between selection
   and its write. Deferred word views preserve the link's scored radius;
   **177 passed / 4 skipped** cover the producer and callers; the original
   failed stream completes **85/85** on the repair.
   All four final lesson arms now complete **160/160**, on one unchanged
   source, with **0 recorded ≥1e8 events** and **0/320 correct answers**.
   All **8/8** final XOR fits pass both the numeric and **4/4** grammar bars;
   the previous recorded numeric counts remain **2/8** and **6/8**.
   Source `eb81e90f4d7f182142942f9abc8e5cd73ec8578297f0fb2e98534ec52ef995dd`
   is frozen for review. Focused validation completes **904/904**:
   **875 passed, 29 expected skips, 0 failures**. The convergence, shared-concept,
   symbolic grammar and forced pending-premise gates pass; the live pending-premise
   variants retain their missing-lesson-artifact skips.
   Alec accepted about **730 affected cases and
   all required gates** in place of the full sweep, retaining all eight XOR
   fits; the added producer certificates and caller coverage bring
   the final collection to **904**. Earlier source panels and every failed
   prefix remain historical. No complete-suite
   sweep or learning improvement beyond these measurements is claimed. No commit or push.
   The momentum-carried projection and a learned top-down pair policy are
   built only if the measurements call for them.
   **Schedule with fewer passes (Alec, 2026-11-08, evening: "lets schedule a
   few fewer passes"; "Merge 3 and 4").** Rounds 3 and 4 are one round, one
   freeze, one review, one landing: `separateChooser` false with the
   one-sided projection and the sides, and generate under compose's proposal
   and credit with `walk_policy` and the verb and modal teachers retired, the
   XOR gate run once, output typing. Round 5 carries the memory
   amendment of 2026-11-08 (shared STM0/STM1 strata, operator concept seats, the collapse
   of the pinned sentence's tree, authority onto the operator concepts'
   trust). The end-of-item measurements (the lesson-13 and XOR runs and the
   metrics above, on round 2's frozen source and the finished tree) are part of
   round 6's freeze and return, not a pass of their own. Every landing is
   folded into the next build text, as round 1's was into round 2's; round 6's
   landing is the same three steps (commit, record, push) on Alec's one line,
   with no further text.
   Reviews accept with recorded warts unless a gate fails (a failed
   `test_convergence` or shared-concept case stops a round); everything else
   is carried into the next text. Three Claude texts remain after round 2's
   return: land 2 and build 3/4; land 3/4 and build 5; land 5 and build 6
   with the measurements.
   **Whole-path return, 2026-11-11:** round 7 landed as `984a75b20e` and
   `a39cbfdef8`. The next round builds one complete-path comparison, one
   reader, adaptive omission, and ordered episodic concept references with
   signed values. Source `58f7881c9dbd` completes all three 160-sentence
   lesson arms and passes the unchanged XOR class gate in **5/10** fits;
   all ten grammar readings recover **4/4** word multisets. The lazy
   conditioner's optimizer-adoption cut is repaired and its native updates
   recorded. Omission never activates in these lesson draws; no acquisition
   or answer improvement is claimed. The full sweep completed all 8,069
   cases; its 125 actionable failures are repaired and the item 1.5 case
   stays carried. Returned source `26dd5a0bda60` finishes with 471 passed
   and ten expected skips in the final affected-file/importer runs. Alec's
   authorized thought-store growth uses the existing sentence structure
   bound: the original 44-row thought now fits, with initial allocation
   unchanged. The measured freeze and the returned repairs are recorded
   separately. The accepted round landed as `b35a4bd316` with landing record
   `d88acd3fab` on 2026-11-12. Its five completion findings are repaired as
   item 2.0, with 42 targeted tests passing on source `a22afbd67871f`;
   the repairs are committed and pushed before item 1.8. Alec replaced this
   repair's sweep with targeted testing and requested the full standard
   suite after item 1.8, without `RUN_SLOW`.
- **2. Housekeeping.** *Moved to item 0.5 on 2026-11-09 (Alec): bounded-test
   batching, D5, the deferred slow marks, the weekly sweep, the per-interval
   logger, the resume certificate and two stale test entries.*
   Delete the radix-backed non-word-major meronomy path (no legacy paths).
   From the 11–11c refactor, verify legacy before deleting (no-legacy rule;
   reasoning methods need Alec's review): `intent_boosts`, WholeSpace `part_chain`,
   `_automatic_synthesize_higher_order` against the pool's discovery path,
   and the META `insert_meta` / `taxonomy_parent` bindings once item 7's
   concept-level taxonomy owns them (`overlap_where_tiling` and
   `where_tiling_for_pass` are already gone). Move the 2.9 MB
   `evaluation-source.tar.gz` out of the repository unless Alec wants it
   in-tree.
   *Added 2026-10-08 (the coverage audit):* delete the legacy typed STM
   `_buffer` that only grows, and the detached role view carried beside the
   native occurrence anchor; rename the native context-read
   `exploration_trial` flag; remove the one-release rename aliases that
   raise with the new name (`Queries.py` `_operation_form`'s retired map,
   `Language.py` `_check_operator_availability`) after the one-release
   compatibility period following 6.2's mechanism landing. Documents that still describe retired or superseded things,
   to correct in one pass: [Params](Params.md) lists `thinkingBudget`
   as live and `selectedThoughtPolicyWeight` as a REINFORCE term (both
   rejected at load; `attentionBudget` is the one budget) and
   [QueryWork](QueryWork.md) still meters `selectedThoughtBudget`;
   `complete.grammar`, `ladder.grammar` and the tied-reconstruction fixture
   still comment `chunk`/`lookup`/`quantize`/`arma`/`what`;
   [GradientFlow](GradientFlow.md) still heads rounds 2d, 2e, 3a and
   the final update "review/uncommitted candidate … no commit or push"
   (all landed at `3de37eef5`, `cce3a4f7b`, `f4a68404e`), gives the serial
   leaf as "meaning poles exchanged for negative activation" (superseded by
   the two lanes) and the round-3 scheme as the
   "narrowing-weighted ceiling" (superseded by the adjacent-word
   wholes); [Architecture](Architecture.md) still offers "each word's
   two symbols … either meaning-pole orientation" at pair search, "has-a …
   narrows (round 3)" in the scheme, and "dimensional lift remains with
   5.5" (FutureWork has it).
   *Added 2026-11-04 (Alec: "I would prefer powers of two for all
   constants"):* every constant set by hand becomes a power of two, and a
   width made by adding such constants stays a sum. One pass over
   `data/*.xml`, `data/model.xsd`, [Params](Params.md) and the
   `TheXMLConfig` defaults:
   - round the free sizes: `maxDocs` (all but `BasicModel_corpus.xml`'s),
     `batchSize` outside the BasicModel family, `verbGroups`,
     `placementNegatives`, `identityOnes` and the `meaning_ones` literal,
     `bindingDepth` in the MM fixtures, fixture `nVectors`, slots and
     `numEpochs`, `stmCapacity`, OutputSpace `nDim` 100, hidden width 20,
     and the cadences of 100 (`checkpointEveryBatches`,
     `branchDiagnosticsEvery`); a fixture whose tests depend on a value
     keeps it with a comment;
   - remove the 152 declared widths that `resolve_identity_layout`
     overwrites at load, so no configuration states a width the code does
     not run, and correct `data/BasicModel.xml`'s header;
   - delete the unread `timeout`; correct `data/model.xsd`'s `mtmPrepend`
     default (eight; the configuration and the code use 64), the stale
     source line references in [Params](Params.md), and the
     `subsymbolicOrder` still described in `data/model.xsd`'s comments and
     [Spaces](Spaces.md);
   - keep what is fixed or is not a size: WholeSpace `nVectors` 9 (the
     eight property rows and the word row), MNIST's 784 inputs and ten
     classes, the ports 8001 and 8002, `recurrenceAngle` 15, `conceptIds`
     20001 and `mathTestDepths`; seeds, versions, rates, weights and ratios
     are outside the pass.
   The measured values listed at item 0 are item 0's decisions.
   **Carried into item 2 on 2026-11-03 (audit of the items above 2.5):**
   (D5, the deferred slow marks, the weekly sweep, the per-interval logger
   and the resume certificate moved to item 0.5 on 2026-11-09.)
   Give every configuration a meaning block,
   then retire `Language.py`'s unpaired form-kernel path and its
   `meaning_layout` guards (58 guards, 20
   configurations without `meaningWidth`). Reword the
   "objectless marker" comment at `ReferenceContext.py` about line 460 when
   parts of speech are learned. In the one-pass document correction, add:
   [FutureWork](FutureWork.md) still keeps `timestamp` and leaves
   situation codes and before/after with 5.5 (retired at `552c626ae0`;
   context wholes and VP phases landed in 3.5), still speaks of a logged
   both-rate monitor that was never built, still says item 2.8 is not landed
   (`a8a09a6acb`), still puts the concept-face negative image's measurement
   "with todo item 9" (now item 0) and calls the 2c follow-up "awaiting
   review" (landed `fcaff2d19`); [SelectedMeaning](SelectedMeaning.md)
   "comparisons remain item 8" and [Params](Params.md) "item 8's
   structural rule" point to item 0 and the Done line; [Architecture](Architecture.md)
   still presents `Priming.propagate`, `geometricSpread`, `_priming_boosts`
   and `_last_priming_diffusion` as built and calls round B "not built" in
   its earlier paragraphs while its as-built paragraph records their
   retirement; and the tied-reconstruction fixture's tense, morphology and
   aspect planning comment (`data/tied_reconstruction_output_benchmark.grammar`
   lines 149–151) goes with the `chunk` and `lookup` comments.
- **1.8. The last architecture round** (Alec, 2026-11-11; decided the same
   day, "It sounds decided as item 1.8"; specified the same evening and
   consolidated on his word,
   [specification](plans/2026-11-11-item1-8-last-architecture-round.md);
   its Codex text follows the whole-path round's landing. The notes below
   are
   the record of the discussion; the specification is the statement).
   "all should go into the last architecture round (which
   should happen before optimization and final housekeeping)": after item
   2.0's whole-path round, before items 1.5 and 1 and the housekeeping of
   item 0.5. Its contents as they stand, each with Alec's words:
   1. *A cheaper forgetting pass.* "Good". `Forgetting.retention_values`
      compares every row with every row; `BasicModel_corpus.xml` holds
      1,048,576 rows
      ([FutureWork](FutureWork.md#a-cheaper-forgetting-pass-running-luminosity-on-ltm-alec-2026-10-24)).
   2. *System 1 as a grammar file.* "System 1/2 is the grammar Vs no grammar
      distinction, which can be broken down even more finely; if we allow
      the interpret operator, we end up with symbols. Otherwise we end up
      with (e.g.) the intersection/union/negation grammar that defines
      animal cognition. So maybe "no grammar" models should just look like
      a grammar file with three operators instead of a hardcoded
      sigma-pi-negation over all three?" Open: Claude's answer of the same
      day (as field operators applied in parallel with no chooser).
      *Alec's reply (2026-11-11, later; "this would be a big change, let's
      discuss"):* "Sigma and pi as grammar operations are serial only
      because they operate on a single concept, but both generalize to
      multiple concepts. My feeling is that, just as the concept of table
      implies the concepts of legs, we probably need to treat serial mode
      as 'all of the conceptual parts and wholes of the given concept'. So
      serial does mean one concept, it means one coherent anchor in the
      conceptual hierarchy. That allows for a much richer single concept
      than a single vector, and it fits pretty well with the existing
      architecture." And: "this would allow single operators that act on a
      field, where serial mode means conceptual part/whole isolation (which
      would happen in virtue of symbolic activation automatically). This
      would mean moving from a point in a conceptual space to a projection
      onto known concepts, but it would unify serial and parallel
      operations nicely." Not decided. Claude's reading of the second
      sentence of the first quotation: serial does *not* mean one concept.
      *Alec, the same evening:* "As an argument against the single vector
      processing as opposed to the field; it's basically a grandmother
      neuron. But being able to use a single vector to stimulate its
      conceptual entailments; reasonable." Claude's reading, given to him:
      the vector stays as the key that addresses and stimulates; the field
      is what operators work on. Open with him: whether a key stimulates
      its entailments through stored references or by inverting the
      operator that made it, as the generate walk does today.
      *Alec, later:* ""One anchor and all concepts entailed by it", since
      part and whole are kind or mereological terms. But whole entails part
      and part does not entail whole, so they are similar mechanism (at
      least at order zero). These entailments are just the existing
      conceptual order." On Claude's point that a projection onto known
      concepts cannot hold what is new: "I think your point about novelty
      might justify it; memory stores known things, not novel things." And
      his question: "does it mean that grammatical operators also have to
      become field operators, not vector operators? That means that verbs
      will operate as matrices over concepts themselves, not matrices over
      conceptual space, which is a very big change." Claude's answer, for
      discussion: no. The entailment operators (sigma, pi, negation) are
      already structures over concepts, the stored definitions. A verb
      stays an operator on conceptual space (`LiftLayer`: log eigenvalue
      gains on an invertible `SigmaLayer`) and acts on concepts through
      the codebook, $M = R\,V\,C$ with $C$ the codes and $R$ the read-back,
      of rank at most the space's width; a free matrix over $N$ concepts
      would cost $N^2$ entries for each verb and give a new concept no row.
      What changes for a verb is that its result is projected onto known
      concepts after each step. `CompoundLayer` already works so ("Select
      the right head's native sigma cases with the left modifier"). One
      precision from the code on entailment: a conjunction entails its
      parts (down the ladder) and a member entails its class (up it), while
      a class's cases are alternatives that a descent chooses among
      (`cs_reverse_presence`), not entailments.
      *Alec's summary (2026-11-11, later still; the working statement):* "we
      can keep doing what we are doing with "unique" conceptual vectors
      (using operators on vectors to create points in conceptual space),
      but then the understanding of those will involve their projection
      onto the entire conceptual field. So in that way, parallel reduces to
      serial at the grammar layer, and then projects back. But animal
      cognition always involves [the] complete conceptual field and the
      three operators that can be applied to that whole field (the fuzzy
      tetralemma logic)." Claude confirmed it, with what it leaves to the
      round: (a) the three operators then exist on both sides, on two
      points in the grammar and over the whole field, and what unifies
      them is that the projection $P$ carries one to the other,
      $P(a \wedge b) = P(a) \cap P(b)$, $P(a \vee b) = P(a) \cup P(b)$,
      $P(\neg a) = \neg P(a)$, which a toy can measure and one side can
      teach the other; (b) "projects back" is done today by the
      reconstruction walk (the generate inverse, whose words become the
      order-zero field), and the projection through the codebook and the
      stored definitions is a second way, so which of the two
      reconstruction costs is to be decided.
      *Alec's conclusion (2026-11-11):* "Ok. So we can distinguish serial
      and parallel learning by the grammar file, and three of those
      operators are field operators, the rest being vector operators."
      Claude's reading: intersection, union and negation take and return
      fields, over the whole field in System 1 and over the projections of
      chosen operands when the grammar applies them; every other operator
      takes and returns points; a file with only the three is animal
      cognition, and enabling operators in the file is the gating of point
      5. With the three on the field only, (a) above no longer needs two
      forms of them to agree; what remains of it is the crossing, a point
      to its field by projection and a field to its key.
   3. *An intuition may answer and reconstruct.* "An intuition is just as we
      discussed; it can do both output and reconstruction". Decided. Today
      `ThoughtAnswer.accepted` rejects evidence of kind `intuition` and an
      intuition's terms are remembered rules.
   4. *Trust.* "Trust of percepts is 1, seeing is believing. After
      word-concepts are turned into object-concepts, though, trust is
      uncertain (0) to start with." Decided; it replaces Claude's question
      about a belief default. The record already has content embedded in a
      supply entering with both lanes at zero
      ([Architecture](Architecture.md), "Evidence belongs to its
      source"); the `<operation>` switch of 2026-09-16 is not built.
   5. *Learning gated by operators.* "We sort of short-circuited
      corpus-gated learning by operator-gated learning. Since that can be
      written into the grammar file, we can just add/enable operations one
      at a time (hopefully without altering the graph?)". Today: two
      authority gates (`interpret`, `clause`) cap trust until measured
      ready and switch no operator off; `GrammarCheckpoint.align` carries a
      checkpoint to a grown grammar by operator name, rebuilding the graph;
      no rule has an enable.
   6. *A query on concepts, and no external `.when` or `.where`.* "the
      query() on concepts needs to be there also. So that means that .when
      does not seem necessary at this point: we store episodic concepts
      (probably by reference) and a single semantic concept (by content),
      and while order is important (and there is a .when baked into the
      semantic concept and possibly the episodic concepts), but there is no
      external .when or .where (it is not perceptual memory)." Today: an
      idea row's stamp is already inside its vector (`RowTime`, "never a
      separate durable time column"); external are the entry's
      `episodic_where` brackets and the relation rows' where and when
      column; an entry is reached only by time (`EntryRead.recall_time`).
      *Alec's reply (2026-11-11, later):* "Ok, keep .when as an external
      model time. But we shouldn't be using it; LTM is content addressable,
      and especially as we are storing concepts, they will have a .when
      muxed into them when they are pulled out of the conceptual
      codebook." And: "I think .when and .where were useful when LTM stored
      percepts, but if we are not storing percepts, we have no need of
      those fields. Do animals need perceptual memory, or is invertible
      concepts enough? I think for now it's enough." Claude's reading: the
      clock stays; no store is addressed by it; the relation rows' where
      and when column goes in this round and the time-only recall gives
      way to the query on concepts. The entry's brackets and pole pair went
      at once, in the whole-path round.
   7. *The loop as an examination phase.* Alec, 2026-11-11: "So serial
      learning is enabled by interpret() and the loop that allows paying
      attention to a sentence multiple times (in order to do serial
      processing). Should we allow the loop in animal cognition, allowing
      animals to pay attention to different aspects of the image even
      though they do not have the ability to symbolically represent those
      attentions? In which case, the loop is best seen as an examination
      phase instead of a symbolic loop." Claude's answer, for discussion:
      yes. Today attention's loop already chooses among six actions that
      the grammar file declares as field rules: three moves (`divide`,
      `descend`, `gloss`: `BracketOperation` actions 0 to 2) and the three
      field operators (`and`, `or`, `not`: actions 3 to 5), which
      `Attention.field_reduce` computes on the two lanes (and: the least
      evidence for with the greatest against; or: the dual; not: the lanes
      exchanged). The open attention pass is the glance before it. What is
      missing: the loop runs only where a configuration has word objects
      (`_has_word_objects`), and its readings go only to compose. So the
      grammar file has three groups (examination; field logic; `interpret`
      and the vector operators), and an animal file is the first two. The
      three field operators exist in several places today (the fold over
      the stored definitions, `field_reduce` over an attended field's
      readings, the binary `intersection` and `union`); the round makes
      them one.
   8. *A linear layer on concepts as the continuous output (proposed).*
      Alec, 2026-11-11: "would it involve a lot to add the linear layer on
      concepts to the XOR as modality with that work, so that we can have
      at least two modalities and keep xor tests without special casing
      them?" Claude's sizing, for his word. Three numeric read-outs exist
      today: a named concept's positive pole (`OutputSpace.read_concepts`
      under `conceptIds`; `XOR_exact.xml`), a linear map on the flattened
      symbolic vectors (`OutputSpace.forward`; the whole-slab XOR files),
      and `_forward_head` on the composed sentence state
      (`XOR_grammar.xml`). One linear map over the bound field's concept
      evidence, both lanes, a weight for each concept row at the
      inventory's fixed capacity, is small, and the named concept is its
      one-hot case. The cost is in the tests moved onto it: moderate for
      the XOR tests that already run on the field or the sentence driver
      (`XOR_grammar`, `XOR_exact`, the grounded test); large for the
      whole-slab XOR proofs (`xor.xml`, `MM_xor.xml`, `XOR_pos.xml`,
      `MM_20M_xor.xml`), which run the older pipeline with no loop and
      include the standing convergence gate. Not sized without a trace:
      what in the field stands for a pair of different concepts in the
      grammar XOR fixture.
   *Still to be settled in the specification* (Claude's defaults, each to
   be flagged for Alec there): the three field operators take the least
   and the greatest on the two lanes everywhere, as `field_reduce` does,
   and not the product and probabilistic sum of today's vector forms; a
   combination of different concepts is held in the field by a concept
   formed on recurrence, a working row before that; operators are enabled
   by named stages in the grammar file that a run selects, over a mask;
   an intuition's conclusion may be said and reversed while its terms stay
   remembered rules; modifier plus noun stays a vector operator; what
   `interpret` gives a new object-concept as its first trust is to be
   traced.
   **Carried from the whole-path review (2026-11-11):** a certificate for
   a new parameter checks that the optimizer holds and steps it, not only
   that a gradient reaches it (the lazily created question conditioner had
   a gradient and was never stepped); the lesson arm watches the float32
   optimizer peaks that the whole-path lessons showed on the chooser's
   `what_projection.weight`.
   **Order, amended (Alec, 2026-11-16):** round C is reviewed and
   committed; the `performance` branch is merged; the rest of this item
   follows (the sentence as one block, then the teacher of the walk of
   attention); the work is pushed; then item 1.7, which was round D
   (specification §43). **The rest, in order (2026-11-16):** every rule is
   written; the five texts for Codex and their returns are listed in
   §67 of the specification (the lanes repair, the tree first, the loop
   of looks with its teacher, a look's forecast and surprise, tables as
   tensors and one graph for a sentence).
   **Expectation, in this item (Alec, 2026-11-17):** "Expectation is
   supposed to be simple: it is not prediction, just an aid to
   understanding and missing data. So let's have expectation work on
   percepts and the field, never on a full sentence. That means never on
   an idea, either, since that is a computed value"; then "Expectation
   is simple, then. intuit over System 1 (S1)", "we are not doing
   predictive processing, we are doing input reconstruction", and "I'd
   like to do expectation in round 1.8". Where an expected idea is
   wanted, it is read from the expected field by an N:1 network, with no
   WFF. That read is taught, an independent network of the form Pi,
   Not, Sigma ("lets expect a conceptual field, and an independent N:1
   pi-not-sigma to reach the idea"), with System 2's idea as its
   target; "The N:1 read can be centroid if we are doing animal
   cognition". intuit is expect's map and, as well, the propagation up
   and down the orders ("lets run the N:N intuit() as the existing
   propagation of N:N expect() *as well as* the (up and down) vertical
   order propagation that is enabled only with symbols and recursive
   cognition"); the downward half is item 1.7's vertical spread.
   Completing every
   operator's field use is future work ("lets add that to future work
   unless it is immediately necessary"; the note for
   `doc/FutureWork.md` is the specification's §68.55). His words are in
   §68.43, §68.45, §68.50, §68.52 and §68.57; the rules in one place
   are §68.53, with two trials; the text for Codex is §68.54, which
   Alec approved on 2026-11-17 ("Yes, as written", §68.58). Codex began
   it and stopped on one question (§59.43), which Alec answered: the
   field read's loss keeps its weight under a key of its own,
   `symbolFieldLossWeight` ("Yes, new key", §68.60, with the text that
   continues §68.54). The build is under way; its return continues §59
   from §59.45.
   Codex stopped again on a question of signs (§59.50 to §59.58): a
   word whose Sigma read is negative is pushed to STM as minus its
   stored code. Reviewed in §68.62 (2026-11-17), where the cause is
   found in rule 1 of §55.1, which removed the step that stored the
   read in the code's unsigned form. Alec, 2026-11-18 (§68.63): "I do
   not know the fix for the current code", and for the final version a
   signed value "should be replaced by the two unsigned symbols or
   percepts that represent the vector: so 0.5 becomes [0, .5] and -0.5
   becomes [.5,0]", on the way up and on the way back. That is rule 3
   of item 1.7, with the way back added (sigma-pi §19.24). The fix for
   the current code is Claude's (§68.63): the word is deposited by the
   leaf law again (`Interpret.activate_code` on the word's own code,
   with the size of the read and the attention walk's pair), the
   committed charts stand in place of §59.53's signed ones, and the
   Sigma's sign fills the lanes with item 1.7. The text for Codex is
   §68.64; Alec carried it. Codex's return is §59.59 to §59.74, not
   yet complete. It stopped on the place and time coordinates inside
   an idea (§59.60); Alec, as Codex records it: "Use sinusoids in 0-1
   range, that's a todo in 1.7 anyway." (§59.62), so stored place and
   time are unsigned since. It stopped again on `bind`'s containment
   test (§59.70) and went on at Alec's "Please continue if that was
   the only stopper" (§59.71). Claude's answer is §68.65: place and
   time stay out of that test, which is the committed rule ("Bands
   are positions"); §59.71 also moved the start of the comparison,
   so that a model with no meaning block binds by its forms, and that
   is to be taken back (*recommended; Alec's word awaited*). Alec,
   2026-11-18: "Direct compare will not capture whether the concept
   is a token of a concept type, whihc is where that information may
   be necessary." The record reads it beside the containment sum (the
   order filter; `ReferenceClose.continuity`). *Open for Alec:* since
   the tree is built over words the menu compares the word, at order
   0, with the held objects (18 candidates offered where HEAD offered
   129 on four test files); what the menu reads for the mention is
   undecided (§68.65, point 3). **Answered by Alec the same day
   (§68.67):** "we should match the object, not the word"; the
   object is matched against types and tokens, with polysemy
   accounted for, by "an algorithm that matches a content against
   Conceptual Space, so that we do this only in one place for all
   clients needing a match"; place and time "are necessary at the
   level of events", and it is "generalization that should remove
   that information, not us by hand outside of the matching
   function". So §68.66 is withdrawn. Codex took §59.71's moved start
   back itself on reading §68.65 (§59.77); the place and time that its
   menu still cuts out by hand go with the one match. Its design, with
   what exists today, is §70. Alec, 2026-11-18 (§70.8): "Yes, as
   proposed", so it is built now, in part B; a word with several
   objects has one picked, "an educated guess from field
   context/activation"; and "Im Ok with only spot testing this
   change". A scratch trial (§70.10): with the menu given the word's
   object and nothing cut by hand the failing test passes, with the
   word it fails, and 144 tests of eleven files pass either way. At
   the menu four mentions in five are phrases; they are left as they
   are in the first text (*Claude's default*). Of the measure, whose
   example is §70.9, Alec: "I have not reviewed the measure above, but
   as long as it aims to fulfill my description, please doc it and we
   will return to it." It is documented as provisional in §70.12, with
   where it meets each part of his description and where it does not
   yet. **To return to: the measure of the one match.** Matched at
   Alec's request with parthood as *The Whole Part* defines it (pages
   184 and 185): the two numbers are that parthood taken both ways,
   and `meet_ratios` returns the values of its table of five cases
   (§70.14); the clipped cosine that `doc/Mereology.md` calls
   parthood is another function and cannot. Of a phrase
   as the mention, Alec (§70.13): it "can still be turned into a
   concept vector and either created or just used as an identity in a
   small pi-not-sigma until it dictates that a symbol should be
   produced for that entity". The close does this today from
   comprehension's value; during the read the menu still tests the
   vector made from the phrase's words, and the first text keeps that
   and has Codex report what the other would take. The text for
   Codex is §70.11, **approved by Alec on 2026-11-18 ("Sure")** to be
   given as it stands; its return continues §59 from its next free
   number. Codex's return is complete at §59.82: its final sweep has
   7,261 cases, 6,929 passed, 330 skipped and none failed, both
   stopping gates pass, and nothing is committed. It records, and does
   not repair, that 139 of 160 lesson fields differ and that the
   lesson's answers are empty. Codex resumed on §70.11 from that
   source. Claude has not reviewed §59.59 to §59.82 as a whole. Alec,
   2026-11-18: "Not now", then "please review it and the next reutrn
   togather." The return on §70.11 came as §59.83 to §59.86: the one
   match is built (`bin/ConceptualMatch.py`), and a single word at the
   menu is tested by its object. Codex committed both returns locally
   under Alec's authorization (`08d1cab2bf`, `31b24db620`; not
   pushed). **Reviewed together in §68.68 (2026-11-18): both
   accepted.** Claude's reruns on the committed source: both stopping
   gates and the five named files, 95 passed; the new certificates,
   33 passed; the 21 files of §68.40, 270 passed and 8 skipped. Not
   rerun: the sweep and the lesson. Part B's first text is returned
   whole; the next text is the repair of the thought replay (§68.31;
   its return is §69).
   **Parthood (§70.14, §70.15).** At Alec's request the measure was
   matched with parthood as *The Whole Part* defines it (pages 184
   and 185), and `doc/Mereology.md` now defines parthood so, in place
   of the clipped cosine: "parthood is not symmetric". Signed values:
   his guess, "per dimension, there is no overlap between a negative
   and positive value", is what the stored two lanes already compute
   (checked, difference 0). Not every inner product is replaced (Alec:
   "it seems like a good fit for MTM and perhaps other places").
   Alec, 2026-11-18 (§70.16): "1 yes", so `Ops._part_kernel`'s scalar
   form becomes this parthood (trial: six definitional tests change
   and nothing else in 58 files); **the text for Codex is §70.17.**
   And "2 agree": MTM's three likeness matchers move to it with item
   1.7.
   **Returned, not yet reviewed.** Codex returned the thought replay
   as §69 and the scalar part function as §59.87 to §59.90, and
   committed both locally (`8b250295e7`, `db95089f90`). Alec,
   2026-11-18: "wait a sec for the codex review." The standing
   documents that still called parthood a clipped cosine
   (`doc/Logic.md`, `doc/Architecture.md`, `doc/STM.md`,
   `doc/FutureWork.md`, and two places of `doc/Mereology.md`) are
   changed to the function as built (§70.19).
   **The loop of looks, begun.** Codex has started the text of
   §60.3 with its preparation (§61) and stopped, as rule 15 says to,
   at a stem with no word offsets: the whole-slab stem of a file
   with field operators only. Claude's answer is §68.69
   (*recommended:* that stem hands over the word intervals it
   already computes, which equal what the search finds; the
   brackets stay, since they are that path's first look). Alec,
   2026-11-18: "the prot, and please doc so I can understand
   laters." So the port is built: **the text for Codex is §68.71.**
   The account in plain terms is §68.70, and in
   [Spaces](Spaces.md#two-stems). Kept for later: on that stem
   a bracket's key is the code of a letter, which matters only if a
   walk ever runs there; Codex reports whether one does.
   **Reviewed (§68.72, 2026-11-18):** the thought replay (§69) and the
   scalar part function (§59.87 to §59.90) are accepted. **The one
   text for Codex to continue is §68.73:** the test rule, the port,
   then the loop of looks. Open for Alec from the review: whether two
   things with no references are the same by their content, and
   whether `ImpenetrableLayer`, which nothing builds, stays.
   **The port is built and reviewed (§61.3 to §61.7; §68.74).** Codex
   stopped again at §61.6: with the old acceptance gone, no act of
   attention can look at a bracket of one known word. Alec,
   2026-11-18: "please authorize it to do a bit more workaround when
   it hits an error (I.e run with a proposed solution, if the
   proposed solution seems in keeping with spec." That is §73, for
   every later text. Codex's solution is taken under it: a focus may
   look at a word (rule 5a of §60.1). **The one text for Codex to go
   on is §68.75.** To return to: the two byte frames of §61.4.
   **A word found by a near miss (Alec, 2026-11-18; §68.76).** "so
   "cat", "caat", "cat " may all be interpreted successfully
   (although with some gradient error on pass/fail that will encourage
   or discourage such near misses in the future)". Not in the rules
   and not in the source: a look is made of whole words, the lookup of
   a form is exact, and a near miss is refused and not charged.
   **Decided (§68.77): item 1.7.** "A look should be a .where that
   returns percept indices. If that is forced to words, then the lexer
   is doing the models job"; a Pi in that neighbourhood finds the
   spellings; "the hit/miss on a word will hopefully feed into the
   width of that pi hypercone". Item 1.8's loop stays on word brackets
   with the exact lookup. **The reply to Codex's stop of §61.6 is the
   text of §68.75.**
   **The loop of looks is returned and reviewed (§61.8 to §61.21;
   §68.78).** The mechanism is accepted: rules 1 to 16 and rule 5a,
   with 213 targeted cases rerun by Claude. One finding waits for
   Alec: with nothing forced and no teacher, a fresh model reads no
   word. The grammar's choice at a word starts one operator cost below
   abstaining (`choose_interpret`), where the starting source had no
   choice at all, so the assessment's rung one went from 8 passes to
   0 and the verbal proof lost its grammar inverse. Simulated with the
   interpretation taken wherever it is offered: every word is read.
   The question, with Claude's recommendation first: no decision to
   abstain (a look that is one known word is interpreted); or the
   decision kept and started even; or kept as built, with reading
   always taught first.
   **Decided (Alec, 2026-11-18; §68.79).** "I do not like the "always
   interpret" option. We should be able to treat a word as a word,
   not as what it represents." A global `interpretBias` "is the
   probability of calling interpret()": "we can set that bias to one
   to force it, but someday we should back it off and only interpret
   words when doing so is valuable based on previous experience."
   It is rule 5b: the call is made with that probability and the
   grammar's chooser decides the remaining share (Claude's reading).
   His first condition, "there is a word there (as indicated by
   inclusion in a meta concept)", is not what the source tests and
   cannot be required on a fresh model, where no word has a
   definition yet; it is recorded for item 1.7 (sigma-pi §19.29).
   **The one text for Codex is §68.80**: the bias, then straight on
   to the forecast of a look (§62.6), returned as §61.22 onward and
   §64. Future work (Alec, 2026-11-18; `doc/FutureWork.md`): "some
   record of having examined a particular place, and feedback about
   the both/neither over symbols so we know if we should attend() by
   zooming in or out", needed once the bias is lowered.
   **The bias and the forecast of a look are returned and reviewed
   (§61.22 to §61.24, §64; §68.81).** Both are accepted as mechanism
   (122 targeted cases pass). A fresh model reads its words; a
   trained one reads almost none: 8 of 184 places after the lesson.
   Teaching moves a whole property's membership from 1 to 0.9998, the
   evidence against is then 0.0002, and the walk's test counts any
   value above zero as a second lane, so the word is never offered.
   Before the loop all 184 were read after the same lesson (§59.74).
   **Recommended to Alec (§68.81):** Codex names what now moves that
   membership and repairs it at the producer, and what "both lanes at
   once" means for graded lanes is answered with item 1.7; the other
   choices are a graded test now, or a floor. Also found: a look at
   several words reads nothing, and by the forecast's rule 3 its
   empty comparison erases
   the surprise of the words it covers; rule 3 is amended (mine), and
   what such a look conceives is item 1.7's rule 32. **The one
   text for Codex is §68.82** (the producer, rule 3, the measurements
   again, then straight on to §65.4, the last text of item 1.8),
   approved by Alec ("Yes, carry it"). Returns: §64.7 onward, then
   §66.
   **The tables are returned and reviewed (§64.7 to §64.11, §66;
   §68.83).** Accepted: one trace, the tables as tensors, exact
   replays (replayed by Claude, 16 of 16 updates equal); the first
   sentence's host reads go from 80,434 to 9,870. Property 0's
   teaching is not new and is by declared costs; Alec, as Codex
   records it: "Keep this baseline as long as we address it later"
   (the graded case is item 1.7's). Part C ends there: the trust
   radius keeps its host calculation as a named exception, and the
   list of capture breaks goes to item 1.5 ("1.5 is the compilation
   to make it fast"). **Found:** two tests of the default tier pass
   or fail by the draw since the loop of looks, one on an exception
   the model raises itself. **The one text for Codex is §68.84**:
   that repair, then the item's one default sweep; and after it,
   with no wait, a test-only prototype of item 1.7's `attend`
   (its text is sigma-pi §19.47). **Codex does not wait on a question (Alec,
   2026-11-18; §75):** "Can it propose the solution itself? its a
   capable mind, and we still have oppostunity to review." It
   proposes, builds its recommendation in a commit of its own, marks
   it provisional and goes on; each text names what comes next.
   **Codex is out of credits (Alec, 2026-11-18)**, so a Claude
   session stands in for it; its text is §68.85. Where Codex stopped:
   the exception of §68.83 is repaired (`1c5dcbb73c`) and the tests
   that read by chance are ported (`1547b3bd7b`); §66.8 and §66.9 are
   written and not committed; **the item's sweep ran and failed, 46
   of 7,325 cases, of which 34 still fail** in twenty test files and
   passed before the loop of looks. So item 1.8 is not at its push
   until those are ported or repaired and the sweep passes.
   **Returned and reviewed (§66.8 to §66.16; §68.86): the item's
   default sweep passes,** 7,336 of 7,336 cases with no failure, in
   2,114 s, on `b656480f7d`. The 34 were ported or repaired; two were
   defects (a presentation with no word did not close; a look's field
   was carried across a packed boundary). Rerun by Claude: 266 passed
   twice in fresh processes, and the lesson's replay 16 of 16 equal.
   **Provisional, for Alec:** a read with no sentence gives the grammar
   its option at each look (three commits that go together); Claude
   recommends keeping it. **Left of item 1.8:** the doc pass (§70.18),
   the PDF, the push.
   **The push (Alec, 2026-11-18): "We should push after 1.8".** After
   Codex's return of §68.84 (the repair and the item's one default
   sweep) is reviewed, the doc pass is applied and the PDF rebuilt.
   Nothing is pushed before that. **Alec, 2026-11-18: "For 1.8, you
   push, and the PDF work is fine."** So Claude pushes `main` of
   `basicmodel` when those steps are done, and tells him first what
   goes; the superproject's pointer stays his.
   **The doc pass, the PDF and the push (Claude, 2026-11-19; 1.8
   §68.87).** The doc-pass lists are applied to the standing
   documents, checked against the source; the PDF is rebuilt; `main`
   is pushed once, through that commit. Left for Alec's word: the
   provisional read with no sentence (§66.12, §68.86).
   **Item 1.8 complete (2026-11-19; 1.8 §68.89, §66.17, §68.90).**
   On Alec's word the read-with-no-sentence repair is removed: a
   parse that gives no WFF outputs nothing. A short series of
   tests passes (the stand-in's and Claude's own); the default
   sweep was not run again. `main` is pushed a second time on his
   word. Left with him: the superproject's pointer. The fast word
   evaluator now lists every item as silent.
   **Commits and the docs folder (Alec, 2026-11-18).** Claude may
   commit its work locally; pushing stays rare. The pending
   documentation pass of the worktree `basicmodel-docs` is on `main`
   (`8cf88d459b`) and the worktree is removed (moved to
   `/tmp/wikioracle-removed-2026-11-18/`). The doc-pass lists kept in
   the specifications can now be applied on `main`. Alec had meant
   the alias `~/github/doc`, a link to the superproject's `doc`
   folder, which he removed himself. Of the pass on `main`: "Keep it"
   (§70.16).
   **At the promotion of item 1.8** (Alec, 2026-11-18: "please make
   sure the doc on main is pushed with the 1.8 promotion."): the docs
   on `main` are pushed with it, and nothing is pushed before. What
   waits, and what is done first, is §70.18: the doc-pass lists of
   the specification are applied to the standing documents, the
   commits ahead of `origin/main` are read once more, and the push
   is on Alec's word. Added by Alec the same day (§70.20): the doc
   changes made in another session go with it, the new section of
   `doc/BasicModel.md` among them, and `BasicModel.pdf` is rebuilt
   (`make doc`), last. This machine has neither pandoc nor xelatex.
   Alec: "Please add the pandoc and typst to requirements, we can
   test later." `requirements.txt` has `pypandoc_binary` and `typst`;
   nothing is installed or tried, and the `doc` target is changed
   when the route is tried.
- **1.75. The part of item 1.5 that makes the tests faster, taken now**
   (Alec, 2026-11-20: "Since 1.5 is likely to speed up testing
   considerably, I think that some portion of that should be done as
   1.75, after this round, and can be revisited if multi-modal code
   changes things or needs new GPU acceleration. That will allow 1.7 to
   proceed with tests much more quickly."). Done after round 4 of item
   1.7 and before its round 5; item 1.5 keeps what is left and looks at
   this part again once item 1.6's second modality is in. What it holds
   is chosen by a measurement of where the tests' time goes, in
   [the item 1.5 plan](plans/2026-11-14-item1-5.md) ("Item 1.75").
- **1.7. `SigmaLayer` and `PiLayer`, and symbols split from concepts**
   (Alec, 2026-11-16: "I think we should position D as item 1.7"; then
   "review and commit round C, and merge with the performance branch. Then
   do everything else in 1.8", and "(and push before moving on to 1.7)").
   Round D of item 1.8 becomes this item, built after item 1.8 is finished
   and pushed: [specification](plans/2026-11-14-sigma-pi-layers.md) §18,
   with §18.22 and §18.23 (symbols without dimensions, split from
   concepts). Its scope includes a carried decision from item 1.8 §64.7. For graded readings, settle
   whether a learned complement counts as simultaneous contrary evidence,
   and test a word with both a strong for lane and a small complement.
   The measured §64 word was “Felix” at (1, 0.00020045042). Item 1.8
   retains exact purity until this is addressed (Alec, 2026-11-18).
   Its form as decided the same night is §18.24: the Pi
   selects and weights the percepts, their vectors are pooled at the basic
   level, and a dense Sigma collects them into concepts; above order zero
   the Sigma reads symbols' activations. §18.25 (Alec, 2026-11-16): signed
   concepts, unsigned percepts; a concept is one signed number in a
   reading, stored as two unsigned lanes, two symbols for each concept.
   **Its rules in one place: §19.21** (2026-11-17), with the text for
   Codex in §19.8. As decided on 2026-11-17: one structure, Pi, Not,
   Sigma, at every order, where a Not entry at the basic level records
   an absence and the operator `not` is put on symbols; MTM nominates
   every row of the Pi, with one concept for each row and both matrices
   thinned; `interpret` makes the meta concept and a blank object; the
   vertical spread, up by fan and down by share; orders that are not
   strictly ramified; and thought reading the Sigma and the Pi
   (`parts`, `wholes`). **Amended 2026-11-17 (§19.22):** the vertical
   spread is what `intuit` runs, with expect's map, when the grammar
   calls it. Open there, with Claude's answer and a proposal to measure
   it: whether the first layer and expect can be one mechanism (Alec:
   "these two N:N computations are the largest in the model"; and "one
   possibility would be to involve the reconstruction to wind up back in
   basic percepts, which is a kind of imagination rooted in the word").
   **A testpoint for it (§19.23):** Alec, 2026-11-17: "please add a
   testpoint that verifies the architecture fills in a blind spot on
   reconstruction (i.e. the pi should generalize of neighboring elements
   to fill in a missing area)." Rule 30 (the spread reaches the first
   layer, and the way back reads what it expects) and certificate 24
   (`TestBlindSpot` in `test/test_testpoint.py`). On the committed
   source nothing is filled: 0 of 5 bytes in every case of the probe.
   It is added with this item and not before (Alec: "With item 1.7").
   §19.21, the rules in one place, now carries rule 30, certificate 24
   and §19.22's amendments of rules 18, 23 and 25.
   **The order of learning (§19.25):** Alec, 2026-11-18: "so we learn
   word generalizations, and we learn object generalizations, and THEN
   we learn word-object meta-concepts, because those are concsistently
   associated. there isnt a statistic that will magically produce
   identity across independent tokens." It is rule 31. The example he
   asked to have in the doc (Helen Keller at the well-house, from
   Walker Percy) is written out there and is now in `doc/Language.md`.
   The two
   testpoints he asked for are certificate 25, `TestCrossModalTie` (a
   word and an object of another modality tied by their consistent
   association), and certificate 26, `TestBellSalivate` (a sequence
   learned: the bell, then the food). Both are added with this item
   (*Claude's default*, as Alec decided for `TestBlindSpot`). Open
   there: how rule 31 meets rule 20's blank object.
   Alec's further remarks of the same day are §19.27: making the object
   for the word is the bootstrap of a model with one modality, and
   stays; two other ways to learn the tie are named and not chosen (a
   relation at zero trust; Pi, Not, Sigma each time, promoted when the
   Pi indicates, with a small MTM with forgetting); an existing object
   is looked for before one is made; and of the blank object, "better
   never to have the blank at all", with both trained and the blank
   removed only if the two vectors come to overlap. Open: what removing
   is, beside concepts kept forever.
   **Looks and spellings (§19.28, rule 32; Alec, 2026-11-18):** a look
   is a `.where` that returns percept indices and is not forced to
   words; a row of the Pi finds the different spellings of a word
   ("caat", "cat") and returns the basic percept for all of them;
   the hit or miss on a word feeds the width of that row's cone. To
   be specified with the item.
   **The largest word that fits (§19.30; Alec, 2026-11-18, an
   idea):** `interpret` is to "take the largest word that fits in
   the .where (even if it is composed of smaller words)". It is the
   dynamic stop of `doc/FutureWork.md` seen from `interpret`; placed
   with rule 32 by Claude, and in no text of item 1.8.
   **A look may carry a content condition (§19.31; Alec,
   2026-11-18, a direction):** "a bit of top-down priming for a
   content match", a wanted whole ("a set of letters and no spaces,
   numbers, or punctuation") with the `.where`, which limits reading
   to words and can be left off. The row exists (WholeSpace's fixed
   word property), and so does the wish (`set_reading`, the frozen
   concept "reading", called by nothing); a look's read takes no
   condition yet, and the stem makes the cut. Placed with rule 32 by
   Claude, and in no text of item 1.8.
   As wires (§19.32; Alec, in discussion): "conceptual space asks
   for the kind of thing it wants (top down activation wires) and
   the perceptual space returns tokens of those types that occur
   within the .where span", a mask on what is active, learned by
   gradient; learning the place the same way is deferred. Open:
   which wires take the gradient, and what a returned token is.
   A request may also name concepts (§19.33): the translation to
   percepts exists (`cs_percept_attribution`, `_passback_scope_ps`),
   outside the loop of looks, with a floor of one half that never
   removes.
   **Decided (Alec, 2026-11-18; §19.44): "Let's move this attention
   and sentence percept bank work to 1.7, then, with the rest of the
   pi-not-sigma work."** So §19.31 to §19.43 are this item's: the
   percept bank of a sentence read once; the first Pi matching it
   through what is active, with no `materialize()`; a look as a
   place and a request for a kind; attention as the filter that
   dims, fed into the gain field with expectation; the selection
   matrix from below ("the selection matrix over percepts that
   light up that concept row": the mask pulled back through the
   matrices of the way up; on a small case it keeps `cat` and drops
   `cot` and `sat`); and the where as a coordinate of the first Pi.
   All of it is in discussion and not yet rules. **The Codex text of
   §19.8 predates it and is not to be carried as it stands.** Six
   points need his word first (§19.44): which wires take the
   selection's gradient; remove or only dim; the exact tests at full
   strength or graded; the two wheres; what moves the look on; who
   changes the request. **Written as tentative rules 33 to 41 with
   seven questions and my recommended answers (§19.46; Alec: "1.7
   spec."), and a text for Codex for a prototype in tests only
   (§19.47), which follows the last text of item 1.8.** Both texts
   approved by Alec ("Yes, carry both"); the prototype returns as
   §20 of that document.
   **His answers (2026-11-18; §19.48).** The gradient lands on "the
   specification of the concepts that create the pullback". A look
   dims, by "the radius of the hypercone (and the magnitude)", with
   no separate floor. The Pi projects "onto two domains (the what
   codebook and a where domain that is just the constant sinusoid)"
   and the two are intersected "across the active percepts"; "the
   output of attention is onto all concepts", so the request is
   `attend`'s and is learned. No second where for arrangement.
   Finding "words in order before doing any grammar on them" is
   taught and "should be a test that gates grammar tests", so the
   reading lesson stays. A word left uninterpreted may enter as a
   literal in the place of a clause, "only if there is a grammatical
   rule to take care of it (like sign())" (rule 42). The fallback if
   the prototype fails: an explicit `.where` and the largest word in
   it. His confirmations: the slow run at item 1.6, part C ended,
   Codex's limit, `interpretBias`. The prototype text §19.47 is
   amended to match. **Then answered (§19.49):** "we want to select
   the most active word-concept to put on the stack during
   interpret(), so it must be exact", so `interpret` takes the most
   active word concept of what comes back, with no draw (Claude's
   reading: with one signed number for a concept, "Felix" at for 1
   and against 0.0002 is then read). And "attend will apply the
   top-down mask on concepts that gets pulled back"; as a proof of
   concept, "pulling back the meta-concept to isolate words in the
   percept codebook". None of the seven questions is open; the rules
   are written when the prototype returns.
   **The prototype returned and is reviewed (Claude, 2026-11-19;
   sigma-pi §20, §19.50, §19.51).** Accepted as a prototype: its nine
   tests pass and its tables reproduce. The look works on the
   lesson's sixteen sentences (the meta concept with a place returns
   the word at the place in 16 of 16; an absent word returns nothing;
   reading in order was learned from the request alone). Rules 33 to
   42 are written as rules in §19.51, with three points for Alec,
   each built as recommended unless he says otherwise: the Sigma is
   passed by its weights in the pullback; the bank holds the rows the
   lookup names (a token's form and its atoms), not bytes; among
   equally active words the largest is taken. Found in the source:
   `PiLayer2` passes no gradient at a membership of exactly one.
   **Alec on these (2026-11-19; §19.52).** He worried that the
   pullback relied on the learning algorithm, and asked whether the
   derivative issue came from preexisting stimulation and whether
   the inverse would do. Checked: yes to both. A look takes no
   derivative; the request is carried down by the way back, the
   rows' own entries (304 of 304 requests exact). "If we have
   equally active words, then we have not identified the correct
   word": the tie rule "first found" is withdrawn. Rules 36 and 37
   are restated in §19.52 for his word. **Alec again, the same day
   (§19.53):** the salience map selects the word level and the
   `.where` selects the word; where several stay selected, one is
   drawn and judged by its outcome; a read that gives no WFF is an
   error to propagate, and teaching starts with minimal sentences.
   **Parthood of forms is to be fixed:** `is` is found in `this` at
   0.82, not 1 (4 of 36 such pairs whole); three ways are measured,
   the store's own `nested` widened is recommended, and the test
   point he asked for (the structure as composed is valid) is a
   certificate. **The read with no sentence:** he finds the repair
   intrusive; proposed in its place, for his word, a test of a word
   among garbage letters (1.8 §68.88). **Alec, then (§19.54):**
   the end marks of a form's code go; a space character is added,
   so that the salience map can select against it from within
   parts; the code is to be explored over a whole's parts of every
   size ("the complete part graph, and the most efficient storage
   of that graph"). Measured: stored as its immediate edges the
   graph over the existing parts is 522,429 edges for 197,014
   forms; with every run of letters a node, two edges a node and
   a whole fixed by its two parts. His part map (all pairs, no
   ends, a longer part where two words would be the same) works:
   1,536 forms share their pairs with another, and a run of 3 or 4
   letters tells 99 in 100 apart; `banana` and `bananana` need 5.
   **Matches by a matrix product, one matrix for each character
   length (Alec; §19.55):** run on the word list, it finds every
   form at every place exactly, and with his part map (617 pairs,
   367 longer parts) the parts matched inside a word give exactly
   that word, 400 of 400; a pair that is no known part leaves no
   trace (`felixx` read as `felix`).
   **Pairs, too, only where needed (Alec; §19.56):** from the
   letters up, 737 parts suffice (392 of the 617 pairs, 297 of 3
   letters, 43 of 4, 5 of 5), by a greedy estimate.
   **Tested on Metal (Alec asked; §19.57):** the parts at every
   place in 0.50 ms a sentence, on the device; the words from the
   parts, 500 of 500 right, 5.9 ms against all 197,014 forms.
   **"Words" presses against the space (Alec):** a Not entry on
   the space, added to rule 35 with a certificate. **A word put
   back wrongly** mostly reads as no word (89 to 99.6 in 100),
   sometimes as another word, sometimes still as the word meant.
   **Reconstruction proceeds from the specific concept (Alec),**
   which associates its part arrays with location arrays.
   **The way back, for generation too (Alec; §19.58):** "the
   whole stores its parts in order, and longest first"; run on
   197,014 forms it puts every form back exactly with 4.3
   entries a form, as fast on Metal as a table of characters.
   Taken, on his word to do what is most efficient; the location
   arrays from the read are withdrawn.
   **The discussion is finished (Alec, 2026-11-19: "this gives us a
   clean perceptual basis spec for 1.7").** What is decided and
   what is open are in one table in §19.59, with the text that
   begins the item in a new conversation; the implementers run in
   Opus.
   **Then (Alec; §19.60):** "we will have to revert to a gradient
   saliency map from the output, since it is input sensitive",
   taken from the open run; "no derivative" is withdrawn and the
   form of rule 36 is open (the plain gradient misses 74 of 304).
   The request is to say where in
   the conceptual tree. A read with no WFF: "we will probably have
   to return to this"; the read with no sentence goes with it
   ("your solution sounds better"; one-word reads by bisecting a
   failed sentence). **The item's text for Codex (§19.8)
   waits: "wait until discussion finishes".**
   **`attend`, tentatively (Alec, 2026-11-18; §19.45):** "we ask for
   percepts by selecting some set of concepts, doing the pull
   (back?), and examining the set of returned word concepts for
   something that we might want to interpret()", the returned set
   being "just conceptual activations from the forward pass after
   the pullback". Written with its steps, what it needs and my
   defaults; to be simulated before it is made into rules.
   **Begun 2026-11-19.** Alec's answers on the open points of §19.59
   are §19.64: the salience map passes each Sigma by its weights; the
   lookup moves to the device in the part map's form, and when is
   Claude's to say (item 1.5); a string read as a word it is not is
   taught when it is observed, by an idempotent round over the
   harness's logs (item 1.6); a read that gives no WFF ends in
   `silence`, S → NULL, a rule in no grammar file; nothing more is
   specified for spellings; and the where flows from a parameter that
   holds the sinusoid. **The rules in one place are §21** (rules 1 to
   48, certificates 1 to 44, with what is Alec's, what is Claude's and
   what is open marked). **The item's text for the implementers is
   §22,** in seven rounds, and it replaces §19.8; the returns are §23.
   Round 2 tries the looks in tests only, before round 6 builds them:
   the form of the where match, and whether reading in order is
   learned with no location concept made, are open until it returns.
   **Added the same day (§19.68 to §19.81).** Two rounds joined the
   seven. *Round 3c, the percept's code* (rules 50 to 57): 64 for the
   parts and 64 for the wholes on two lanes, for and against; the
   parts built up from all zeros, the wholes down from all ones, on
   the part catalog; the Pi selecting for and against; a scalar by
   the same mechanism. It was tried in tests first (§23.3b). *Round
   6b, a fragment read at one time by the basic level* (rule 58):
   the words of `the cat` read together into one idea that System 2
   teaches, close to the idea composed from them one by one, as an
   option for the grammar; no form or symbol is made for it. The rules now run to 58 and the certificates to
   50.
- **1.6. The number modality, the math corpus, and the felt world that
   combines them** (Alec, 2026-11-18: "Then 1.6 becomes the number
   modality, the math corpus (it needs more development), and the work
   you called out as "unnumbered: felt world for the math corpus" which
   combines both."). "1.6 is the proving ground for what it is doing"
   (Alec, 2026-11-18, of item 1.7's architecture). After item 1.7 and
   before item 1.5. The work that
   had this number until then, computation by wiring, is the first part
   of item 1.5. **Added 2026-11-19 (Alec, sigma-pi §19.88):** "We should try
   to teach with reconstruction only, and try just reading without
   comprehension as a task prior to reading with comprehension. Add
   this detail to 1.6". So this item has a task before reading with
   comprehension: the model reads a sentence and puts it back, with
   no parse kept and no question answered, and the reconstruction's
   error is the only teacher of where a percept is. Item 1.7's round
   6 builds the where so; round 2 of that item measured the
   reconstruction alone at 4 held-out orders of 6.
   **Added 2026-11-20 (Alec, sigma-pi §19.99):** "Let's add a test for
   sequence learning in the sense of "singing a song over the output".
   For 1.6, counting would be a good test of that". A taught sequence
   is produced over the output from its first item, each item from the
   ones before it with no input between; here it is counting (the
   testpoint `TestSingASong`). And this item's lessons are ordered by
   successive approximation: "we can't teach focus directly, but we
   can reward behaviors that require a successively narrow focus".
   **Added 2026-11-20 (Alec):** "1.6 is mostly about multi-modal and
   good testing." Its corpus describes a scene and uses no
   anthropomorphic verb: not "one runs to two from the left" but "one
   is moving toward two from the left", a 1 moving toward a 2 over a
   background of 0s. Frames 16 and 20 of the design are restated so.
   **At its submission the slow tier (`RUN_SLOW`) is run
   once, overnight,** and not before: the architecture changes no more
   after this item (Alec, 2026-11-18; 1.8 §72). **Begun 2026-11-18**
   (Alec: "a detailed spec for how we can teach a synthetic math corpus
   in which human language describes numerical (continuous) percepts"):
   [the design](plans/2026-11-18-item1-6-felt-math-world.md), with
   what exists (§2), continuous perception at places on three axes
   (§3), what a second modality changes in word and concept processing
   (§4), the corpus and its grades (§5), a word of place (§6, not
   designed), certificates F1 to F12 (§7) and its questions for Alec
   (§9). **His first answers, the same day (§11):** "Add 2d intensity
   images also"; intensities "Varied from the start"; for a word of
   place, "A small image in which the numbers are vertically arranged
   as described"; and the gendered words are dropped for "words like
   even and odd, and other simple math concepts like basic operations".
   **His second answers (§12):** three axes are kept, an image being
   the plane at z = 0; "a level is a 0-1 float always"; the bracketing
   of a felt value "is adaptive anyway"; and the corpus is a proof of
   concept of learning language from "simple sentence and small image
   pairs", in which a thing is "a continguous shape constituted by 1's"
   that has a size and moves ("which was faster, the walk or the
   run?"). A first sketch of that world, its words and its grades is
   §12.5, and what each round waits on is §12.6. Open (§12.7): when a
   varying reading is the same percept again, where the vocabulary of
   mathematics is taken from, and what a numeral names. **His third
   answers (§13):** "percepts are discriminants", so two readings are
   one percept when their parts and wholes are the same, "just as we do
   today"; the sketch stands; and "we need a set of training examples,
   maybe 100 per, categorized according to which english grammatical
   rules are used within that sentence", the cast "single digits", with
   "the evens and the odds", and "generalization and inference to work
   over that space before we begin paying for computer time on a remote
   machine". The 26 categories, each with its example, its question and
   what it waits on, are §13.5; a row and its validity §13.6; what is
   measured §13.7. Open: §13.8. The generator and its frozen rows touch
   no model source and can be given to Codex before item 1.7 lands.
   **The text for Codex is §14.8** (Alec, 2026-11-18: "Yes, write it"):
   `bin/felt_world_corpus.py`, the recipes and frozen rows under
   `data/felt/`, and `test/test_felt_corpus.py`, from the world of
   §14.1, the predicates of §14.2 and the frames of §14.4; it returns
   as §15. The model's part of the item has no text and waits for item
   1.7. What the record held before it:
   1. *The number modality.* Built in item 1.8: a number is a level
      percept at its place; one felt value may accompany a sentence, at
      one fixed place (`bin/ScalarPercepts.py`); the XOR proofs and
      `data/MM_image.xml` read numbers at places on one axis and on two
      with field operators only (`NumericPercepts`). Listed as
      remaining in
      [FutureWork](FutureWork.md#numeric-input-as-a-modality-xor-as-a-system-1-model-and-one-model-class-alec-2026-11-11-to-specify):
      a second axis of place for attention; learning on images, first
      as a measurement; grey levels; one model class; and the
      whole-slab body of the other numeric configurations.
   2. *The math corpus.* "it needs more development"; which
      development, he has not yet said. As built by item 0.5
      ([spec](plans/2026-11-09-item0-5-test-development.md) §7 and
      §17): nine frozen grades under `data/math/` with a manifest
      (`bin/math_chain_corpus.py`), its numerals opaque words and
      nothing numeric reaching the learner (Alec, 2026-10-07); its bar
      is undeclared. Its learning measurement at the trained
      checkpoint, with the four protocol corrections, stays in item 0,
      and the comparison of compiled and eager runs on its rows in
      item 1.
   3. *The felt world for the math corpus,* which combines both (Alec,
      2026-11-18): "for the math learning corpus, we should make a
      model test that matches words to objects, where objects are
      feeling-intensities within a 3-D space. So a sentence like "one
      was on top of two", even though it doesnt make a lot of sense,
      will still describe a world that is phenomenally accessible to
      the model, an in which the feeling of one is actually above the
      feeling of two. That also forma a basis for the identity tests
      that we are talking about now. Maybe girls are even and boys are
      odd, so we can map a gendered language onto numerals." His words
      in full, what exists and what it needs:
      [specification](plans/2026-11-14-sigma-pi-layers.md) §19.26.
      Not yet specified: the feeling modality with three place axes,
      the scenes generated with the corpus's sentences, and the tests
      (a numeral's word tied to its felt intensity, a word of place
      tied to the relation of two places, two words with one
      perceptual basis tied to one object).
   **Silent outputs, mined and bisected (Alec, 2026-11-19; placed
   here by Claude, 1.8 §68.89).** "the training harness can mine
   the logs for instances of silent output and do bisection to
   teach words recognition without sentence parses." A parse that
   gives no WFF outputs nothing since item 1.8; the harness finds
   those sentences in its logs, halves each until a word stands
   alone, and teaches each piece with the word it holds as the
   expected output. Not designed.
   **A word read as another word, taught when it is observed (Alec,
   2026-11-19; sigma-pi §19.64, point 3; placed here by Claude).**
   "We can teach that difference when we observe it in the output.
   so just an idempotent teaching round, gathered from the training
   harness logs". Nothing is added to a form's code for it. The
   harness logs each word that is put back as another word (`felixx`
   read as `felix`); a round gathers those cases and admits each
   string as a form of its own; run again, it changes nothing. The
   store's side is item 1.7's (sigma-pi §21, rule 46 and certificate
   28). Not designed: the log's row, and when the round runs.
   **A test left to chance, to return to here (Alec, 2026-11-19;
   1.8 §66.17, §68.91).** `test_interleave_lifecycle.py` failed
   in 3 of 3 fresh processes on the swept source, since it counted
   the untrained chooser's draw, and the default sweeps passed
   it. It now reads in source order. To find: why the sweeps
   passed it, and whether other tests of the default tier hide
   the same way. Also here until it reads sentences: the fast
   word evaluator's `score` command lists every item as silent
   and is kept out of every gate and run.
   See the item's plan, "Suggestions for the item, from `suggestions.txt`"
   (doc/plans/2026-11-18-item1-6-felt-math-world.md), for a first spatial
   milestone, corpus changes, and measures of learning on the local machine.

### Live items 1.5 to 0, as they stood in todo.md

- **1.5. Computation by wiring; then lookups as one matmul, whole/part
   folds as scatters, the percept pass per sentence** (Alec,
   2026-10-30; and 2026-11-18: "Please move all 1.6 work into 1.5").
   **Current priority (Alec, 2026-11-26).** Pause the other 1.5
   optimizations. Focus only on making exploration faster in ordinary
   training and testing. Convert each sentence to percepts once. Share
   the opening bank and greedy prefix across its alternatives. Keep each
   path's evolving parse and provisional memory private, with one clock
   per sentence and only the winner committed.

   On local branch `item1-5`, `7e6e34dafd` still walks the alternatives
   sequentially. The first experiment is four sentences with three
   explorations each. After the greedy pass, run the twelve exploration
   lanes together wherever practical. Compare against the same three
   departures run serially, checking choices, winners, committed state
   and elapsed training time including packing and winner selection.
   Group changes before the 12-update replay and the sampled/evaluation
   reading lesson. If separate lanes require substantial bookkeeping or
   give no speedup, revert the path-as-batch experiment and keep the
   simpler shared work. The existing one-versus-three timings measure
   sequential walks. The next comparison needs the same path count.

   **Open, 2026-11-24 UTC.** Local main shares sentence percept reads and
   the opening forecast. Each stream selects its winning exploration at
   sentence close. The ordinary input ladder still reads learned span
   endpoints on the host and prevents strict whole-sentence capture.
   The [integration handoff](plans/2026-11-24-item1-5-handoff.md)
   records real training speed, restored accuracy, gradient checks and
   remaining work shared with 1.6.
   After item 1.6 and before item 1, so the compiler work compiles what
   this builds under `fullgraph`, and before item 0. It has two parts.
   *Claude's default:* the wiring first, as it stood before the lookups
   when it was an item of its own.
   **The radix trie as a tensor (Alec's question, 2026-11-18;
   [plan](plans/2026-11-14-item1-5.md)).** Percepts are found today
   by a Python trie on the host. Two forms that are wires: the trie
   as a transition table run for every start place at once (results
   equal; a trial agreed at 173 of 173 places), which is this item's;
   and no trie in recognition, the Pi lighting a word from its atoms,
   which is item 1.7's first layer.
   **Decided (Alec, 2026-11-18: "Ok."):** the radix store and its
   codes stay as they are; the overlap tree (a whole stored as the
   overlap of its two parts one letter shorter) is exact but is not
   built now; the lookup moves to a compiled table on the device in
   this item, with equal results.
   **Amended 2026-11-19 (sigma-pi §19.53 to §19.57; §19.64, point
   2).** Exact parthood between forms is wanted after all, so in item
   1.7 a form's code becomes the set of its parts (no end marks, the
   space as a character, a part only where two forms would be the
   same). Alec, of finding the word from its parts without a product
   against every form: "we might as well move to the powerset version
   we developed if that will resolve this. Up to you when we do
   that." It does, with the set of parts used as the form's address:
   0.76 ms a sentence for both stages on Metal against 6.25 ms, and
   the same answers at 500 of 500. *Placed here by Claude:* the lookup
   moves to the device in this item in that form (the parts at every
   place by one product for each character length, then the word by
   one probe), with results equal to item 1.7's; and with it the
   table of a form's longest parts in order, for the way back
   (sigma-pi §21, rules 44 and 45).
   Alec, 2026-11-18: "1.5 is the compilation to make it fast". The
   attention and the percept bank of one sentence, placed here for
   part of that day, are item 1.7's (sigma-pi §19.44).
   **Added 2026-11-19 (Alec: "Please add this to the 1.5 work ...
   most seem like things you can do in parallel with 1.6").** Five
   steps from the last profile, in the plan under "Where a sentence's
   time goes": score the candidate paths as one batch or score fewer
   (45% of a sentence); look up a sentence's percepts once; no read of
   a tensor on the host inside a sentence (24,547 a sentence); no hook
   on every saved tensor; and the recurrence over the words as the
   floor, moved by batch size and bf16. *Claude's reading:* the first,
   third and fourth can be built beside item 1.6. The profile is taken
   again on the source item 1.7 leaves before any text is written.
   The largest speed-up Alec names, the words of a fragment such as
   "the cat" read at one time by an S1 that S2 has taught, is made an
   option at the end of item 1.7 (sigma-pi §21, rule 58) and measured
   here.
   **Part 1, computation by wiring, not by moving data** (item 1.6
   until 2026-11-18; Alec, 2026-11-17: "can you investigate the
   possibility of pushing the architecture in this zero-data movement
   direction? It's 1.6 so that subsequent optimization becomes easier,
   with fewer cpu or python data reads/writes"). Investigated, measured
   and re-scoped on 2026-11-17; nothing built:
   [the document](plans/2026-11-17-item1-6-zero-data-movement.md).
   Decided that day (its section 9): the trace is the one form of a
   sentence's tree, built with part C of item 1.8 (1.8 §65.5); MTM and
   the rows it promotes into are one table; the choice among paths is
   left as it is. **Measured** (its section 3; one worked document of
   the math chain, eager, on the CPU): for one sentence 10.0 million
   Python calls, 1.1 million calls into the tensor library and 24,547
   host reads of a tensor; scoring the paths is 45% of a sentence, the
   read 26%, its optimizer step 12%, the kept close 9%. The walk of the
   pinned tree is four fifths of the `.tolist()` reads (step A). MTM is
   at most 5% of a sentence and the kept rows' write 0.3%. **Revised
   steps, decided** (Alec, 2026-11-17: "Can we update and finish open
   questions in 1.6 as a prelude to GPU optimization?", then "the four
   points are good to go with your recommendations"): step B, MTM as
   rows, is designed (an entry is a row, and whatever else it carries
   is a pair); step C, the block commit, is withdrawn as wrong on the
   facts; step D, a meaning is made once, and step E, what is cut from
   the graph is cut where it is written, are added; the hook on every
   saved tensor is measured in item 1 first; the reverse's zero buffers
   and the derived forms are part 2's, with their numbers. Its section
   7 is what GPU optimization receives, with Alec's point of the same
   day that a sentence's percepts are looked up once, at its start
   (part 2's percept pass).
   **Part 2, lookups, folds and the percept pass** (item 1.5 as it
   was). *Amended on 2026-11-18 for the loop of looks (1.8 §74;
   Alec: "Please see the sentence optimization of 1.5 and decide how
   this attend logic should be split"):* a sentence is perceived once
   into two arrays, the percept indices and their `.where`, and a look
   is a mask on the second that returns indices into the first; item
   1.8 builds the loop, item 1.7 what a look is, and this item what a
   look costs. Alec's note: the mereology is built once for the
   sentence, and the Pi creates the basic percepts at each look, in
   dependence on its `.where`. So below, the percept pass is over the
   sentence's parts and wholes and not its words; "the forecast scored for all words at
   once" is withdrawn in that form, since a forecast is made at its
   look; and a known word's percept comes from the Pi's row of item
   1.7, not from its identity row.
   From the as-built cost reading of 2026-10-30: the
   derived concept forms become a tensor kept current incrementally
   (device CSR definitions, one segmented scatter per order, a validity
   mask); content-based lookups over absolute truth concepts in LTM and
   conceptual space become one matmul of the cue against the content
   matrix with top-k under the work meter (`relation_evidence`, taxonomy
   evidence, witnessed rows; LTM1's index maintained on `add`;
   `MediumTermMemory.match_candidate` and `FamiliarTier.witnessed_match`
   become one product of the support centroid against the masked codebook
   rows, decided 2026-10-30; the cued LTM0 read, `cued_rows`, already
   scores its `ltmWindow` rows in one batched product since item 2.5,
   though it still gathers them row by row and ranks them on the host);
   the percept pass runs once per sentence over all words with gradients,
   since the percept depends on neither grammar state nor the expectation
   (Alec again on 2026-11-17, for the GPU: "compute perceptual features
   for the whole sentence at the start (IOW, do all codebook lookups by
   building an array of percept indices for the .what and an array of
   .where with the same indices)"; measured then at 1,924 lookups of
   codebook rows for a sentence, 63 of them of a whole table: the
   wiring document, section 7),
   with ragged parts in place of the 64-wide padding, a known word's
   percept read from its identity row's `amax` instead of recomposed from
   its atoms, and the forecast scored for all words at once; an explore
   path resumes from `SentenceFork`'s state at its departure word instead
   of rerunning the shared prefix. Grammar operations themselves stay
   serial; tokenization and new-type admission stay on the host, cached
   per type. Gates:
   `fullgraph` compile of every rewritten brick, byte-equal or
   certified-equal results against the current path, one before/after
   timing on lesson-13 and the math chain with the July baseline retaken,
   allocation reads with no resize. Decided 2026-10-30: one Codex round
   for these four changes. The design, step by step, is in
   [FutureWork](FutureWork.md#throughput-levers-for-the-serial-loop)
   ("Lookups, folds and the percept pass").
   **Carried into item 1.5 on 2026-11-03 (audit of items 2.8 and 2.7):**
   while rewriting the derived forms, the atom property fill (2.8's
   O1): `refresh_property_fills` computes a pair atom's fill from its two
   characters without the marker byte, and gives length and mint atoms a
   fill of ones (`bin/WordIdentity.py` still passes `bytes_of(atom[1])`);
   a certificate on `refresh_property_fills`, `containment_audit` unchanged.
   **Carried from item 2.0 (basic percepts, decided 2026-11-04, built in
   round 5):** item 1.5 adopts the basic-percept feature layout, in which an
   order-0 definition holds only the columns `2*bp + pole`: at order 0 its
   fold becomes a gather and its derive scatters move into perception,
   keeping the `working[:n0]` contract that `cs_read_memberships` and
   `_compose_order0` share. It also ports
   `test/test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row`,
   red since round 5: it still expects separate native part and property
   columns where one basic-percept reference is now written.
   **Order (Alec, 2026-11-03):** item 1.5 follows items 2.0 and 2 and is built
   on main. The 2026-10-30 sandbox branch `item1-5-lookups-and-folds`
   (`9151b6f10b`, about 1,400 lines over `EntryRead`, `LTM1`, `Layers`,
   `MediumTermMemory` and `Spaces`) is a reference, ported onto main after
   2.0's move of the three stores into `bin/Memory.py`, not merged.
   **Added 2026-11-19 (Alec: "lets explore implementing the explore paths
   as separate batches which share data structures and sequencing in
   1.5"):** a sentence's explore paths become rows of the batch over one
   shared reading and one sequence of steps, in place of running one
   after another with the state saved and put back; with it,
   `explorePaths` above one is measured against a larger batch. The
   design and its order of work are in
   [the item 1.5 plan](plans/2026-11-14-item1-5.md) ("The explore
   paths as rows of the batch").
- **1.4. Verbs in the field of concepts** (after items 1.6 and 1.5;
   [the item 1.4 plan](plans/2026-11-23-item1-4-verbs-in-the-field.md)).
   **Added 2026-11-24 (Alec: "Please add this work to 1.4. I am not clear
   that the intensity XOR will work until it is completed."):** evidence
   at order zero is positive, with no evidence against and no negation
   there; the Not is one weight that exchanges a concept's two lanes; a
   Sigma of any order ranges over the basic percepts and symbols of every
   order beneath it (the plan's §10). It removes what tells against a row
   in the first Pi and the Not of order zero, which both stopping gates
   run through, so it begins with the proof that signed weights on
   positive rows give every concept the value it has. Waiting on it: the
   XOR of intensities, the name formed by MTM, and the order of two words
   through grammar training (item 1.6 plan, §44).
   **Begun 2026-11-26 (Claude, at Alec's word; plan §11).** The proof is
   measured by `bin/PositiveEvidence.py` and
   `test/test_item14_positive_evidence.py`, with nothing of the model
   changed. It holds on both stopping gates: the XOR gate's against lanes
   are the conjunction's row at weight 1, to 2e-16, and the other gate
   reads no basic percept. It does not hold on the reading lesson: a row
   told against because an occurrence has its parts and lies outside its
   whole has no positive row to carry it (2,757 of 3,468 occurrences), so
   on text the removal changes behaviour. Waiting on Alec: whether the
   conjunction's -1 is written at nomination or learned, and what becomes
   of "outside a whole" (plan §11.4). Nothing is removed until then.
   **First part of the removal built, 2026-11-26 (local branch `item1-4`,
   uncommitted; plan §12 and §13).** Alec decided: MTM writes the link and
   its value is learned from zero; "outside a whole" goes; a weight below
   zero at order zero is one percept telling against a concept at its
   place. The first Pi now reads for alone, and `link_holders` writes each
   held row's concept an entry at zero on its holder's row. Both stopping
   gates pass; 1,083 scoped tests show no failure beyond the 38 that fail
   on `ffbed4fd5b` itself; eleven certificates of the old lane are
   rewritten. Left: taking the dead lane, the stored against entries and
   the Not of order zero out of the code (after item 1.5's shared read
   settles), whether value levels are linked, refinement's use of "both",
   the rules' text in the sigma-pi plan and [Logic](Logic.md), and a
   default sweep.
   **Second part built the same day (plan §14).** The dead lane's code and
   the Not of order zero are out (the second lane of a percept's value
   keeps its place, empty); refinement's "both" is read one order up, across
   the concepts a concept reads (`both_above`); the documents are updated.
   The levels of intensity are linear and fixed by default
   (`architecture.intensityLevelsFixed`, for bfloat16 and so that a
   remembered value does not drift; plan §14.6); with it false they learn as
   a linear self-organizing map, very slowly (plan §14.3), and two
   certificates of item 1.6 that held a level to the mean of what fell to it
   are rewritten. Open for Alec: the map's two rates; the Sigma
   links between neighbouring levels, which were not asked for; `read_word`
   still reads a signed match. The complete
   default sweep is to run before the branch is pushed.
   **Documentation to do with it (Alec, 2026-11-24):** the import of
   positive concepts at the various orders. A concept with no exchange of
   its lanes is a monotone function of positive evidence; say in
   [Logic](Logic.md), [Orders](Orders.md) and
   [Philosophy](Philosophy.md) what that means at order zero (a
   percept present), at order one (a noun, a name, a kind) and above, what
   a mind of positive concepts alone can and cannot think, and how it
   bears on the named variants ([FutureWork](FutureWork.md), Animus
   and Augustine).
- **1. Compiler work.** *Built by Claude, side by side with item 0.5 on
   Codex (Alec, 2026-11-09): the generated math corpus, its generator and its
   frozen manifest are item 0.5's; the comparison of compiled and eager runs
   on that corpus stays here.* *Added 2026-10-20:* the generated math suite under
   `RUN_SLOW` (item 3, moved from 3.5 on 2026-10-21) is the compiler test at scale — compiled and eager
   runs of the identical corpus, seed and budget agree in actions,
   probabilities, costs and parameter updates.
   **Carried from the item 3.0 landing (2026-10-26):** four runtime failure groups: `test/test_objective_conflicts_slow.py::test_native_production_objective_measurements` at the 28 GiB guard, with the remaining allocation unattributed after the Householder and candidate-context repairs; `test/test_testpoint.py::TestGPUDevicePlacement::test_model_on_gpu` at its 120 s deadline; `TestServerQueries::test_demo_queries`, `::test_response_format` and `::test_response_length_bounded` in `test/test_testpoint.py` at their 30 s request deadlines; and `test/test_math_thinking_training.py::test_successor_is_learned_as_a_verb`, `test/test_reconstruction_roundtrip.py::test_grammar_output_loss_not_silent_zero` and `::test_mm20m_grammar_free_derivation_roundtrip` at their 45-minute caps with half work. These are runtime failures, separate from item 0's learning records.
   *Added 2026-10-04:* the open read of 6.8-1 made the
   whole forward 2.6× slower (.141 s against .055 s on the XOR fixture);
   Alec: "the slowdown will have to be addressed." *2026-10-09:* the
   math-chain campaign ran at 1.5–2.5 s per sentence under two workers
   (episodes of 32 thought steps included), a week for thirty trainings;
   Alec deferred the thinking measurement "until we further optimize
   things" — this item's measurement now includes a thought episode's
   step and the per-sentence time at the math corpus. *2026-10-10:* profile
   the rising per-presentation runtime in the 6.1 curriculum (the final
   relation-answer stage took 1,001.95 seconds for 64 presentations plus
   its initial/final measurements);
   compare like-for-like work before assigning a cause. The decoder's two-word
   walk (a known two-word unit decoded by one undo, the tree inside it
   carried by the root rather than walked) may pay part of it back. The throughput levers for the serial loop — batch
   across sentences, the known-word lookup concession, subsampled
   reconstruction, closing the host islands and moving
   the sparse stores to device CSR, single passes — are listed in
   [FutureWork](FutureWork.md#throughput-levers-for-the-serial-loop)
   and belong here (Alec, 2026-09-25); the `serial` flag and the
   `subsymbolicOrder`/`symbolicOrder` knobs those levers named are rejected
   at load since 6.8, so the levers apply to the one interleaved loop.
   Forward chooser split/lift once per slot; backward's
   launch-bound kernel count per brick; B24 brick +25% against pre-ladder.
   The comparison reader of round 2e is a `functional_call` copy of the
   whole answer path; share its structure when
   the answer path grows.
   Exit: sentences/s and peak memory at the run's batch and brick size against
   the July baseline, re-taken on `99207a3` since the tower fold ladders are
   gone. New eager hot spots from 11–11c to measure and vectorise: the
   per-row loop in `cs_read_memberships`, the per-row composition in
   `_compose_order0`, the edge loops in `_reverse_field`, the per-reading
   descent in `refine_over_collected`, candidate growth in
   `_prepare_part_learning`, and the per-stream loops in
   `FamiliarTier.observe_field`.
   The explicit slow reconstruction-cache probe fails its nonzero
   compose-gradient assertion on the current baseline; the cache claim remains
   unverified by that probe.
   *2026-10-30 (Alec):* the as-built cost reading and the lookups, folds and
   percept-pass work are item 1.5,
   which runs before this item so the compiler work compiles what 1.5
   builds under `fullgraph`.
   *Generate walk (Alec, 2026-11-11: "You can do the generate walk
   first"):* `Generative.MenuStore` keeps grammar-inverse menus between
   the walks of a batch, reused when the input content and the parameter
   versions match, and the explore walk is skipped when no stream has a
   legal departure. Certified in verify mode (each reuse rebuilt and each
   skipped walk run, all equal) and by same updates (ten seeded lesson
   sentences: identical parameters and costs after every batch, in 27%
   less time). It exists only as uncommitted work on the local branch
   `item1-generate-walk` of the first machine, cut from `3c9399199d`
   before round 7 and the whole-path round of item 2.0. Next candidates:
   batching inside a built menu, proposals through the same store, and
   the walk under `fullgraph`.
   **Carried from item 2.0's whole-path round (2026-11-11):** a thought
   store may now grow at run time (Alec authorized it), exactly and rarely,
   within its `4 * (W + O)` bound; a growth changes a buffer's shape, so
   under compilation it happens outside the compiled regions, or the
   recompilation it causes is accepted.
- **Item 0 binding evidence:** `test_live_pending_premise_closes_in_both_orders`
  retains its strict arrival assertions verbatim, with `open_names=('x',)`
  and no named closure forcing. Its `pending_premise_learning` artifact is
  unregistered. Measure live closure at the trained checkpoint; correct-or-open
  is the step-1b mechanism gate. Arrival menus remain excluded from delayed
  mention training and answer supervision.
  Under the grammar-only contract, x is a site only when its reading is
  order 1; this certificate gets no fixture order, so learning that a name
  reads at order 1 is part of what it must show.
   **Carried into item 1 on 2026-11-03 (audit of items 3.5 and 2.7):** the
   per-sentence timing includes the nested-admission path (word-form
   reindex, hash and trie rebuild after an admitted suffix) in one bounded
   lesson smoke, and the stationary-walk stop and the eager decoder's
   selected-inverse path; exercise the three `torch._assert_async` sites
   (`ChunkJournal.py`, `Expectation.py`) once on MPS before the run. The
   generated math suite under `RUN_SLOW` was already here.
- **0.5. Test development pass** (Alec, 2026-11-09; built by Codex, side by
   side with item 1, which Claude builds;
   [spec](plans/2026-11-09-item0-5-test-development.md)). On the day item 0
   starts, every measurement it reads has a harness that has run once on a
   small model; the training corpus is frozen data under one manifest, with
   every held-out set listed and certified disjoint; each lesson has a
   corpus, a criterion and one recorded small-model run; and the weekly tier
   has been green once on the hash that holds this item and item 1. The item
   changes no learner mechanism and makes no learning claim at scale; a
   mechanism defect it finds keeps its strict assertion and goes to Alec.
   File ownership against item 1 is in spec §3.

   **2026-11-13 continuation on testing:** the remaining template games,
   stage/criterion/operator evaluation and H9/H10 task scaffolding are built;
   bounded rehearsals and strict nulls are recorded in spec §20. Exit still
   requires byte-exact checkpoint/update restoration, the missing historical
   whole-pullback capture, runtime control definitions and lesson/study bars,
   and the final joint item-1 default/weekly exit. H1 now has a complete
   256/16-versus-64/4 comparison on one source (spec §20.8); the retained
   default is 256/16, with no fixture-reuse speedup claim. Game learning
   remains stopped at its two signal nulls.

   The work:
   (a) the existing test and harness work from items 2, 1 and 0 (spec §5;
   the text moved from item 2 is below, unchanged);
   (b) transformation games (spec §6; Alec's note is below): one corpus per
   grammar rule, a template tier and a written tier, both frozen;
   acquisition, vocabulary transfer, context transfer and composition, each
   against an ablated arm and a shuffled-pairing arm; the unary-chain game
   supplies the evidence the note asks for;
   (c) the synthetic math problems corpus as frozen graded rows, with
   protocol corrections 1 and 2 and the measurement declared again after
   item 2.0 (spec §7); item 1's comparison of compiled and eager runs reads
   these rows;
   (d) the training corpus under `data/training-corpus.json` (spec §8),
   staged by Alec's word of 2026-11-09: words only, then simple sentences
   and the synthetic games, then math, each stage read operator by operator
   (spec §9.1) and its model saved with tests (spec §9.2); only after that
   gate the million-plus sentence corpora (the developmental assembly,
   NanoChat's base corpus), prepared here as data; with one list of
   held-out sets and a disjointness certificate;
   (e) lessons and registered artifacts (spec §9): stage data for the ten
   lesson rows, a criterion per stage, and artifact restoration into the
   trained-tier tests;
   (f) the harnesses item 0 reads (spec §10): item 8's 256-task generator
   and arms, 6.5's six gates, the held-out cost set, the per-interval
   logger and the resume certificate.

   Exit: spec §12. The five decisions of spec §14 were answered on
   2026-11-09 (spec §15); no round text is written until he asks.

   **Test development feedback (Alec, 2026-11-09):** teach individual
   operations through focused transformation games. For example, give a
   sentence and ask for its negation, with a corpus devoted to that one
   transformation: the learner recognizes the “negation game” as an
   exercise. Current LLMs can generate synthetic examples for these
   single-rule corpora; many other grammatical operations may be taught
   this way. Test both acquisition within the game and transfer to new
   contexts, since learning in one context may carry over in part. As
   these lessons supply evidence, revisit and relax arbitrary structural
   restrictions such as prohibiting several unary operations at the same
   location; the increasing repeat penalty is provisional, not a claim
   that legitimate unary chains are impossible.

   **Moved from item 2 on 2026-11-09 (text unchanged):**
   Restore safe, faster bounded-test batching: the default
   256-case/16-file and 32-case/4-file workers exceeded the 8 GiB cap, and
   eight-case/one-file batches pass the same selection; do not raise caps or
   omit cases; reconsider fixture reuse
   if useful ([evidence](DevelopmentHistory.md#forward-owned-lexical-forms-september-19)):
   one preallocated model per compatible fixture family in a bounded worker,
   with a verified complete reset, for the tests that need no fresh
   construction, timed against fresh construction before any speedup is claimed.
   D5: regenerate the two
   decomposition snapshots under `output/item3-5-identity-initial/` from
   seeds 4199722147 and 2428440506 under the 256-wide layout, keep both old
   hashes (`dbd7315f...`, `f1d141ab...`) beside the new ones with the
   generator command and the layout difference, then run both
   `test_math_chain_decomposition` arms at budgets 512 and 768 with the
   original assertions; no legacy loader, no seed replaced. Then the slow
   marks deferred at the 2.8 and 2.9 freezes (26 core, 4 tower-projection,
   3 native packing, both decomposition arms, and
   `test_review17_measurement_wiring_on_one_ordinary_batch`) run once under
   `RUN_SLOW`: in item 2.5's `make test_all` if
   that freeze comes first, else here; `make test_all` green once on one
   hash after D5 (the four item-1 runtime groups excepted) and the default
   sweep green at the ten-worker 28 GiB ceiling, the `XOR_grammar.xml` gate
   decided first (item 2.0). The training harness's per-interval logger
   (carried from item 4): reconstruction loss, expectation discrepancy, LTM
   occupancy and forgetting passes, the categorical-discrimination index on
   the fixed four XOR and 68 FineWeb probes (CP, within and between
   distances, no thresholds) with the both-rate per bracket level, the
   luminosity of provisioned truths at each forgetting pass, the held-out
   generic-subject probe ("cats breathe" and "a cat is an animal" parse
   relative, "this cat breathes" and "the cat is black" absolute), one
   fixed reconstruction sample per interval, and order-0 inventory rows
   against `nVectors`, reusing
   captured readings and budgeting their collection cost; item 0's stop
   conditions read it; the 11–11c pool, raise, refinement and
   unknown-per-order counters are retired with learned orders. Extend
   `test/test_corpus_checkpoint.py` to run the real `runBatch` on a small
   config and assert the resumed next batch's losses and kept programs
   byte-identical, with explicit checks on LTM0 `surprise`, `access_count`
   and `forget_*`, the relevance weights, the comparison reader's weights
   and the form centroids; exit for the resume certificate: one `train.py`
   command on a small config with `checkpointEveryBatches` set, re-run to
   resume.
   Remove the deleted
   `test_forced_ordinary_bound_declaratives_open_no_episode` entry from
   `DEFAULT_CERTIFICATES` (`test/slow_tests.py` line 100). Confirm the 7.5
   shared-operator report assertion is gone under 6.9's one-owner rule and
   strike the FutureWork sentence that defers its verification.

- **0. The full training session**: the long FineWeb run, only with items 11–1.5
   done and item 1 measured. Expectation on in `model.xml` and
   `BasicModel.xml` (`sentenceExpectation`). Stop on rising
   expectation discrepancy, a reconstruction regression against the
   re-taken baseline, a forgetting pass deleting protected rows, a
   provisional-pool exhaustion warning, or order-0 inventory approaching
   `nVectors`. *Baselines:* the `99207a3` and reviewed-9b reconstruction
   figures predate round 3a, after which identified-word reconstruction is
   an exact zero audit; re-baseline reconstruction on the accepted
   `f4a68404e` source before the run (the Done lines of items 8 and 5 cite the old figures
   until then). *Condition (confirmed by Alec):* the
   run trains by expectation and output along one path, with expectation's
   sources live and reconstruction retired to an audit, only where item 3's
   audit reads zero at the run's width; where it does not, the run keeps
   reconstruction as a loss for that part. Expectation across the
   architecture itself landed as item 4 (`fdeef42b8`); the invertible path,
   the width and the audit this condition reads were item 3's from
   2026-10-19 and are the capacity-run line carried below. At the trained checkpoint, resume the
   deferred **MM_math_chain learning measurement** under all four corrections:
   one-step problems first; worked steps scored as intermediate answers;
   an episode that binds a found candidate; and budget in steps, with each
   query's record charge bounded separately. The stopped campaign supplies
   no learning claim; preserve it and declare the corrected measurement
   before running. Bars:
   held-out binding 9 of 10, the chain bar, the expectation-only and
   zero-budget controls;
   the four corrections are unimplemented and precede the measurement;
   re-declare it after item 2.0 retires the decomposition teacher and
   rebooks the choice reward to output.

   **Item 3.0 step-3 companion residual: 6/8, learning-side.** The length-1 question of document 1 in both arms keeps
   an explored `whole` over greedy `surface` under the existing R+A rule.
   On placement's untouched archive, the same greedy program already has
   coverage 1.0; its explored path also has 1.0 and loses on iterations.
   The chunk path's explored coverage 0.75 wins. No reader repair or forced
   keep is warranted by the decided replay criterion. Preserve seed
   **4199722147**, original snapshot and budget **512** for this residual;
   length 2 remains correct in both arms at seed **2428440506**, budget
   **768**. All 32 premise readings remain complete. Measure the chooser
   and reconstruction objective's learned choice at item 0, without
   substituting the standing existence proof for companion accuracy.

   **Slow-budget note (Alec, 2026-10-26):** the entries below retain their
   measured historical budgets. Optional RUN_SLOW repeated work now defaults
   to half; `RUN_SLOW_FRACTION=1` restores the full budgets for a like-for-like
   replay.
   No earlier result is relabeled as a half-budget measurement. The companions'
   512/768 work budgets and one-shot contracts are unchanged; runtime stops
   remain separate from learning failures.

   **Item 3.0 landing — 2026-10-26:** Claude's review approves step 5 `c4a1d92e3f`, serializer `caece43fc0`, sweep repair `03fbbf2a7d9c`, final model repair `6fe273d2f12a7`, and test budgets `ee7ca4e920bc`. The learning records stay at item 0 at their listed budgets; the four runtime failure groups move to item 1 as named below; the unforced 1-to-2 raise certificate for “Felix is a cat. Ciao is a cat.” moves to item 3's corpus round. Learning claims use `RUN_SLOW_FRACTION=1`; a half-work result states its work. No RUN_SLOW before November at the earliest.

   **Item 3.0 sweep learning remainder:** blind `MM_20M_xor` reconstruction
   is 0.0 exact match after **160 epochs**, seed **3145616037**, in the
   initial slow sweep on `caece43fc0`. The repair slow run on `03fbbf2a7d9c`
   also records **0.0 exact match at 160 epochs**, seed **3510236551**.
   Location recovery is 0.0625 and 0.125 respectively; these are separate
   fresh initializations, not a learning improvement claim. The assertion
   stays unchanged; retain both original-budget records with the 6/8 companions for item 0.

   The separate scaffold exact-reconstruction gate also records **0.0 exact
   match at 160 epochs**, seed **1506395199**, with location recovery **1/3**
   on `03fbbf2a7d9c`. Its original 1.0 assertion remains, and its 160-epoch record stays at item 0;
   it is separate from the blind measurements above.

   The two ordinary generation-gradient variants also remain learning-side:
   one requested two-stream batch at learning rate .001, 48 trips per output
   walk, zero emitted concepts and zero conditioner gradient on the recorded
   initializations. The detailed replay chooses `not` / `null` and reaches
   no emitting continuation. Keep both nonzero-gradient assertions unchanged
   for item 0; no forced choice or additional training was supplied.

   Three native answer-gradient variants also have a zero conditioner gradient
   after one requested two-row batch at .001 and 48 trips. The recorded replay
   emits one unconditioned role, leaving the conditioned role pending; it keeps
   the strict assertions and original budgets for item 0.
   The fourth native variant passes after its form-credit
   and kept-field observer port on the original initialization; it is not
   another zero-total-gradient result.

   The full slow sweep's fresh decomposition initialization also retains a
   two-step proof failure: 0/2 final bindings at **571/768 work**, reproduced
   on both `03fbbf2a7d9c` and pre-repair `caece43fc0`; the one-step case is
   2/2 at **246/512**. Keep the strict proof and saved states for item 0.
   This is separate from the passing original-snapshot
   arms and their 6/8 companions. The existing expected arithmetic learning
   gates also remain below .9: direct arithmetic **.3125/.1290 at 40 epochs**,
   dependency chains **.5234/.1953 at 30 epochs** (train/test).
   Serial supervised output reaches **.78125 best accuracy at 40 epochs**,
   learning rate .001, batch 8, after the completed-field observer port;
   the original 1.0 bar remains.
   The cold-boundary corpus reaches its unchanged learning assertion after
   the equality-close repair: both boundary weights stay **-4.0 at six
   epochs**, batch 4, 48 rows, learning rate .001, boundary rate 40/update
   interval 16.

   The supplied neighboring-sentence answer gate also remains open: on the
   frozen sweep source `03fbbf2a7d9c`, the future-label arm's surface error is
   **.0860481 before and .0965504 after 30 updates**, batch 2, learning rate
   .001. Its original 10% improvement bar, answer masks and gradient checks
   remain unchanged. The past-label
   arm records **.0799439251 → .0799437165** at the same budget against the
   unchanged 10% improvement bar; its masks and gradient checks also pass.

   The supplementary three-epoch XOR reconstruction draw (seed 1646304963)
   records 0 exact match against the unchanged .5 bar. A source comparison
   at the original sweep's recorded seed 2197030207 reproduces its .5 exact
   match and identical losses; neither record replaces the other or supplies
   a selected acceptance seed.

   **6.1's learning gates** also remain at item 0's checkpoint: supplied
   answers and asked parts; free reading in the lesson's order on permuted
   fields, with order-lesson CE near zero; and the context stages:
   parallel documents sharing vocabulary, each question answerable only from
   its own; a document resumed after an interruption; a fact from one
   document needed in another; a word whose referent differs between
   documents; expectation from the current document's history.
   The support mechanism and the first two generated-curriculum stages are
   accepted; these learning claims are not.

   **Item 4's learning gates** remain open: graded versus shuffled
   (seed 23's shuffled control has lower R); global derivation optimality;
   per-level residual/collapse measurements and exposure; learned repeated-cat
   recognition. The standing all-level XOR/math regression comparison is
   a gate of the preceding outstanding-gates round, not a deferred
   learning campaign. The original failed native run is not
   regraded or extended. The mechanism and controlled carrier certificates
   do not satisfy these learning gates.

   **Iteration 1 learning carry, 2026-10-24:** swap/verb source `65451694e`
   passes source order 8/8 but fails both original CE bars; the original
   lower and definite budgets each pass bracket/occurrence 1/1. Description
   stops at answers 1/4 with brackets 4/4. Carry description → adjective →
   definite-adjective → final evaluation → copular identity here, with
   the existing budgets, gates and prerequisites. Nothing after the stop
   ran; no copular prerequisite is lifted or artifact promoted.
   On modal source `01effdcaf9`, both the free-leaf and affix corpora
   complete their fixed 24 training examples / 36 paired updates, then
   fail held-out answers 0/4 and required timeline histories 0/4. Carry
   the modal corpus's learning here; no seed, budget, forcing or menu rescue.
   Each form's 21 retained sentence-batch keep records contain no
   alternative timeline and no fired teacher; no trial applied `adverb`
   to the marker. Start the learning remainder from operator choice.
   Before retrying, carry the teacher pairing by branch point rather
   than model-close tick; the corpus has no after-observation target for
   the verb teacher by design.
   No learning claim is made.

   **Iteration 2 learning carry, 2026-10-24:** on repaired source
   `04c1132797`, all six marker placements complete the original
   three epochs / 18 training documents / 162 paired updates and six
   held-out documents. Every placement scores answers 0/6 and exact
   reverses 0/6; selection counts are 6/6 only for the unmarked form,
   0/6 otherwise. Retained selected-output flags occur in three forms,
   but no event retains a case value; two forms show limited verb-teacher
   activity, and none shows a modal teacher or alternative timeline.
   Carry the learned phase selection, phase answers, surface recovery and
   word-class generalization here, with the original budgets and gates,
   no new draw or extra training. The mechanism tests do not satisfy
   these learning gates.

   **Item 6's learning gates** remain prerequisites at this checkpoint:
   stage 4e's ≥90% free-bracket and ≥95% NP-occurrence gates, zero duplicate
   writes, every lowered NP yielding one order-1 occurrence (description stamp
   reported, not required), and no unlowered-description occurrence; then
   balanced binding by group and document-close row pairs, with mint/bind
   fixed semantics and same-kind revision; VP temporal-difference prediction;
   direct-operand word classes; and deferred order-2 kind recovery. The stage 4e
   candidate and measurement exist, but its gate fails.
   The VP teacher mechanism is published at `f306fc28aa`; order-2 kinds are
   not built. Do not start another native
   identity run before stage 4e passes. Retain stored-idea recovery beyond two
   words, order CE, relation trace 2/4, free readback 0/4, MM grammar recovery,
   compiled retention and c's marker-headed `five` as named gates, not
   completed learning. The paired zero-cost invariant is repaired by Step 0.

   **Prepared in item 0.5 (2026-11-09):** the test development feedback
   that stood here moved to item 0.5 with its work. Built there before this
   run ([spec](plans/2026-11-09-item0-5-test-development.md) §10): the
   math-chain corpus with corrections 1 and 2 and its measurement declared
   again (correction 4 is mechanism and is not item 0.5's); item 8's
   256-task generator, its hashes, the arms and the routing catalogs; 6.5's
   harness; the held-out cost set; the per-interval logger and the resume
   certificate; the training-corpus manifest. The runs, the registrations
   at the trained checkpoint, the capacity search and the measured
   constants stay here. Item 0 opens with the three analyzable stages of
   the training corpus at the long run's configuration, saves a model after
   each, and starts the million-plus sentence corpus only after Alec has
   read the operator table (Alec, 2026-11-09; item 0.5 spec §8, §9.1,
   §15; decided by Alec the same day, spec §15.1: small models for the
   experiments of item 0.5, the full size for the long run only, though
   the math corpus may need it).

Everything that is decided in direction but not on this path is in
[FutureWork](FutureWork.md).

   **Decisions collected at the start of item 0 (Alec, 2026-11-03):** the
   evaluation harnesses below, declared before the run: item 8's causal
   study (its 256-task held-out generator, the arms and the thresholds),
   6.5's six learning gates (no harness yet), and the held-out set behind
   the held-out cost metric of item 2.0's round; until then item 2.0
   reports acquisition, answers and each chooser's departure win rate.
   Added 2026-11-04, under Alec's powers-of-two rule, the measured
   constants:
   - `batchSize` 28, the memory operating point: 16 or 32;
   - `maxDocs` 11,207, chosen for 1,000,014 sentences: 8,192 or 16,384;
   - the concept width 1256 (`meaningWidth` 896 beside the 360-wide
     percept event): 1024, with `meaningWidth` 664, or with 512 and a
     reserve whose effect on `pole_stride` is unverified;
   - `conceptLayers` 5: 4 or 8;
   - the thresholds at 3 (`conceptMintCount`, `raiseCount`,
     `mereologyRefinePatience`): 2 or 4.

   **Carried into item 0 on 2026-11-03, from the items above 2.5 (audit):**
   - *From 9.* At the trained checkpoint, once `checkpoint_readiness` reads
     at least 1,000,000 completed FineWeb training sentences
     (`training_state.fineweb_training_progress`; fused-AMP updates
     conservatively uncounted), register it under `fineweb` in
     `test/fixtures/phase-artifacts.json` (path, SHA-256, config, producing
     receipt);
     a missing exposure skips, never passes. Then run the five `fineweb`
     certificates: the 56 held-out wording, generation and rereading
     assertions on the loaded model, prediction's benefit from context,
     predictive thought at gain 1 against gain 0 (less work at no worse
     MSE/BCE), the depth-3 end state and truthset provisioning; the
     tiny-run nulls and the old context-free and assertion-thought scores
     are invalid controls.
   - *From 9 and 8, the causal gate, declared before the run.*
     Independently trained arms (learned thought, direct answer, equal
     actual compute, reconstruction only) from seeds 0, 1 and 2 for 10,000
     updates after qualification, on the 256 held-out tasks whose generator
     and hashes are frozen before training (the generator is not yet
     written);
     gate per seed on accuracy at matched actual work or work at matched
     accuracy; ordered prediction must beat the shuffled and context-free
     controls without worsening reconstruction or discrimination against
     reconstruction-only, with the reconstruction and discrimination control
     re-declared after item 2.0's order-0 reconstruction; report warmed
     sentences/s, operation counts and memory per arm (20 warm-up and 100
     timed updates). Item 8's routing-coverage comparison over declared
     nested catalogs that include one opaque candidate: the sentence opaque
     share must decline strictly in every seed; a catalog without an opaque
     candidate is a descriptive zero, not a decline.
   - *From 9, 6.5's learning gates, which have no measurement harness yet:*
     held-out anaphora against the rule-based binding of `206a0146` at
     matched exposure, verb reuse (one event kind takes the same change
     column across held-out streams), prediction against the detached-target
     control, shuffled order, renamed vocabulary, and mint-after-"a" and
     bind-after-"the" against a shuffled-determiner control (the choice in
     the softmax, not a column's allocation); seeds 0, 1 and 2, never tuned;
     declare and build the harness before the run;
     `test/test_sentence_references.py` is mechanism-only and does not
     satisfy them. Also the concept-face negative image's joint and
     useful-query learning (round 2 `3de37eef5`) at the checkpoint.
   - *From 5.5.* The occurrence learning measurements with the
     six-placement retry: past and future of one event told apart in
     generation, the progressive not entailing completion, a marker
     recovered in reconstruction; recorded, not tuned (the
     irregular-as-stem-plus-past reading is withdrawn).
   - *From 4.* Score the frozen NanoChat next-word manifest with
     `bin/eval_nanochat_grammar.py` at the trained checkpoint: top-1,
     reciprocal rank, intact minus shuffled; recorded, not tuned.
   - *From 2.7, before the run.* The capacity search and selected-prefix run
     of `test/test_corpus_large_slow.py` (B28/W256,
     `data/corpus-capacity-grid.json`, the fitting target `nVectors` and
     `ltmCapacity` 262,144, relations and witnesses 65,536), recording item
     2.7's exits: loader time, admission and occupancy,
     high-water marks, sentences/s, `maxDocs`, `shardDir`, streams,
     `RowTime` range and ladder periods, the D ~ c k log N derivation with
     its assumptions, and the native R == 0 audit at the chosen width (the
     invertible-path condition above reads this line); the PS1024 to
     PS33792 points of item 2.7 are development evidence, not a fit.
     The companion residual's "original snapshot" is item 2's D5
     regeneration from the recorded seed. Before the mass, count and
     kind-verb measurements: at sentence 132 of the final curriculum, where
     `Queries._existing_symbol` finds a query's concept reference
     unavailable, capture `Language.program_meaning`'s native ids and
     selected leaf indices, the `GrammaticalThoughtRegistry._payload`
     reference, `ConceptAllocator.placement` with its retired members and
     MTM's symbol-to-transient identity map, and repair at the producer.
   - *From 2.7, the learning remainders.* The unforced 1-to-2 raise,
     "Felix is a cat. Ciao is a cat.", as lesson-13's relations criterion
     (160 sentences, 8 epochs, 80 supplied answers, `raiseCount` 3, nothing
     forced; latest 0 to 1 of 80 bound, cat at order 1), and the mass,
     count and kind-verb phases of the final curriculum (288 training and
     16 held-out sentences, four epochs per phase; proper phase passed, mass
     unmet at 131 of 304);
     order-2 kinds and kind-level verbs (3.5, 3, 2.7) measured unforced
     on the lesson protocol, mechanism first if the checkpoint shows none;
     silence as the taught target on rows with no supplied answer (3.0 step
     4): the model learns to say nothing unless asked, measured at the
     checkpoint, no forced mask.
   - *From 3.5.* The branch-point pairing is retired with the
     modal teacher (item 2.0); the modal-corpus retry starts from
     operator choice under participation training.

### Done list removed from todo.md on 2026-11-26

*Alec, 2026-11-26: "Please delete 1.6 and above as done." Every entry of the Done list as it stood, items 1.6 and above, kept verbatim.*

- **Item 1.6 — the number modality, the math corpus, and the felt world that combines them:** integrated (Alec, 2026-11-26); [design](plans/2026-11-18-item1-6-felt-math-world.md). Residue, status not confirmed: the one overnight `RUN_SLOW` run set for its submission; the harness mining its logs for silent outputs and for a word read as another word (neither designed); why the default sweeps passed `test/test_interleave_lifecycle.py` when it failed alone ([the block as it stood](DevelopmentHistory.md#integrated-items-25-to-16-as-they-stood-in-todomd)).
- **Item 1.7 — `SigmaLayer` and `PiLayer`, and symbols split from concepts:** integrated (Alec, 2026-11-26); [specification](plans/2026-11-14-sigma-pi-layers.md) §18.
- **Item 1.75 — the part of item 1.5 that makes the tests faster, taken now:** integrated (Alec, 2026-11-26); [the item 1.5 plan](plans/2026-11-14-item1-5.md) ("Item 1.75").
- **Item 1.8 — the last architecture round:** complete 2026-11-19 and pushed; no landing hash recorded, integrated (Alec, 2026-11-26); [specification](plans/2026-11-11-item1-8-last-architecture-round.md).
- **Item 2 — housekeeping:** integrated (Alec, 2026-11-26); its test and harness work moved to item 0.5 on 2026-11-09. Residue, status not confirmed: the legacy deletions (the radix-backed non-word-major meronomy path, `intent_boosts`, `part_chain`, the typed STM `_buffer`, the rename aliases); the one-pass correction of documents that describe retired things; the powers-of-two pass over `data/*.xml`; a meaning block for every configuration, then retiring `Language.py`'s unpaired form-kernel path ([the block as it stood](DevelopmentHistory.md#integrated-items-25-to-16-as-they-stood-in-todomd)).
- **Item 2.0 — one gradient, one choice mechanism, memory as the state of affairs, behavioral chains and basic percepts:** round 7 landed as `984a75b20e` and `a39cbfdef8`; the whole-path round landed as `b35a4bd316` with landing record `d88acd3fab` (2026-11-12).
- **Item 2.3 — trust in operations:** integrated (Alec, 2026-11-26).
- **Item 2.5 — thinking: derivation over chunks with trust, and `intuit`:** round A landed at `3718d9d0dc` (2026-10-29); the item landed at `1e02cebf5d` (WikiOracle `b126029`, 2026-11-06).

*2026-11-03: the eight items above 2.5 (9, 8, 5.5, 4, 3.5, 2.9, 2.8, 2.7) collapsed to the lines below after an audit of each one's remaining work; their blocks as they stood are in [DevelopmentHistory](DevelopmentHistory.md#work-recorded-in-todomd-until-2026-11-03), and the remaining work is in items 2.0, 2, 1.5, 1 and 0 under "Carried ... 2026-11-03".*

- **Item 2.7 — corpus at target size (formerly item 3):** closed code-complete at review freeze `472f5000850e`, commit `0aedc06414` (2026-10-29), reviewed by Claude; corpus assembly and curriculum loader, stage-preserving packing and checkpoint continuation, the familiar tier, reading-mode identities, generation repair with certified completion, the definition fold, percept expectation with its in-loop update, admission and retrieval. Thirteen scoped groups 2,545 passed, 57 skipped; lesson-13 160/160, ideas pass, relations fail, answers 0/80 with text; final curriculum 131/304, proper phase passed, mass incomplete, count and kind-verb not reached. No landing, no acquisition claim. D7 went to [FutureWork](FutureWork.md#the-residual-split-a-stopped-subtraction-or-the-correction-alec-2026-10-29); D8 landed in 2.5 round A `3718d9d0dc`; the non-finite kernel gradient and the order/parts disagreement were repaired in 2.5's completion round. Remaining work moved to items 2.0 (the supplied `yes` admission, the MTM-value recurrence), 2 (D5's snapshot regeneration, the slow suite and sweep, one comment), 1.5 (the `WordIdentity` atom fill) and 0 (the capacity search and the run with its recorded exits and the width audit, the sentence-132 capture, the unforced 1-to-2 raise and the mass, count and kind-verb measurements, the silence gate).

- **Item 2.8 — order-0 concepts sorted over two form subspaces, one call:** landing `a8a09a6acb` (WikiOracle `a992602`), reviewed source `555e63e2f521`, Claude LAND through Alec; identity join and property soft meet in one percept vector (`propertyDim` 256, `propertyOnes` 1), tower-1 content derived from byte membership, `_interval`, `centroid_forms` and `SymbolCentroid.place` retired with absence certificates, the four-report `containment_audit`; 20-case certificate and native lesson green, 2,053 scoped passes, 48 skips, 0 failures, with the two failed freezes `be4514dedea8` and `70f21c757906` preserved. D5's snapshot regeneration and the two decomposition slow arms are item 2; the O1 atom fill is item 1.5.

- **Item 2.9 — priming as location-based diffusion:** landing `17e9454763` on frozen source `b031b11ee9f2`, Claude LAND through Alec with the R1/R2 document repairs; 1,873 gate passes, 47 existing skips, slow marks deferred, no `[V,V]` tensor. Superseded at item 2.5 round B (Alec, 2026-10-28/30): `Priming.propagate`, the radius controller, `_priming_boosts` and the `geometric*` knobs retired after the paired lesson-13 measurement (both spreads 0/12, no learning claim); the `_priming_seen` ledger, the class centroids with the 15-degree contiguity read and `part_edges` stay.

- **Item 3.5 closed — iteration 2 (lines 6–7) and the mechanism landing:** reviewed source `75345f891e` (on main as `2067dbc25a` after the 2026-10-25 history rewrite), published main `4f136f76fa`, WikiOracle `69f82f5f77`, Claude LAND through Alec: document covers and `.when` segments in batch-indexed stores, centroids on form with operations on meaning, VP phases as case selection, direct-operand occupations, the presentation boundary with persistent cover concepts, capacities 18/20/20 by derivation; 588 scoped passes and 41 existing skips on the second repair after the one end-of-3.5 sweep (6,504 cases, 35 repaired). With iteration 1 and the step 1–2 lines below, 3.5 is a mechanism landing with no learning claim: the six phase placements (0/6), line 3's rerun and the modal corpus are item 0's; order-2 kinds, kind-level verbs and the math suite went to items 0 and 1; the root-position and event-history follow-ups are in FutureWork.

- **Item 5.5 — occurrence, tense, aspect, the preposition and surface form:** mechanism landing `552c626ae0`: the clock column, `Clause.eternal` and the analyzer retired; order is the stream's history with store-order recency (`get_stm_chain`) and the stable address re-witnessed in place; `surface` as absorb and transpose with exact inverses; the preposition idempotent; nested sub-word wholes by the type criterion; the clock-, offset-, chain-, interleaving-, preposition- and surface-inverse certificates (`test/test_item55_mechanisms.py`). Relative `.when` and the address upsert landed earlier at `f4a68404e`; the remainder landed through item 3.5 — documents and streams as context wholes with a cover row, VP phases as case selection under one grammar across six marker placements, direct-operand classes (`f306fc28aa`, `75345f891e`/origin `2067dbc25a`). No learning claim: the occurrence learning measurements and the six placements (0/6) are item 0's.

- **Item 4 — run harness and resume test:** the frozen next-word gate `bin/eval_nanochat_grammar.py` (document-held-out manifest, intact and shuffled-prefix controls, top-1 and reciprocal rank from the model's own word expectation bank; `test/test_nanochat_grammar_eval.py`), last repaired at `7285b382e6`; mid-epoch resume certified by `test/test_corpus_checkpoint.py` and `test_checkpoint_restores_optimizer_counters_and_rng` — corpus cursor, batch and step counters, RNG, optimizer state, STM slabs, MTM candidates and mints, priming ledgers, percept arrival state, LTM0 addresses and LTM1 relations restored and the same epoch continued on the remaining rows (`0aedc06414`); forgetting passes log deletions by origin and the value cut-off (`b1a19c5b8`). Carried: scoring the trained checkpoint (item 0); the per-interval logger set and the real-`runBatch` byte-identical one-command certificate (item 2). The number's earlier use, the expectation mechanism, has its own lines below.

- **Item 8 — structural preference and routing evidence:** landing `206a01468d` (WikiOracle `58cc110`) from handoff `074093ec52`; exact-score tie preference for the structural face inside the ordinary MLP (`GrammarPreference.py`), committed-program routing counts and the nested-catalog coverage comparison (`GrammarEvidence.py`), the arithmetic-isolation and renamed-vocabulary probes, and the review's corrected marginal and complete-position overlap; one source-matched 4,980-case sweep (4,653 passed, 326 skipped, one XFAIL) with reconstruction exactly matching reviewed 9b. **No learned-utility claim:** the predeclared causal study, the declining-opaque-share comparison (no catalog yet holds an opaque candidate) and qualified-arm throughput are item 0; the `XOR_grammar.xml` reconstruction gate goes to item 2.0 with the decomposition teacher's retirement, its class-accuracy sibling passing; the MM-grammar gate is the seed-41001 regression Alec fixed on 2026-11-03, with the twenty-start panel as the initialization-sensitivity record; item 6's carried learning findings are at item 0.

- **Item 9 — mature-checkpoint learning evidence:** `8bc710a` (exposure counter `training_state.fineweb_training_progress`, the 1,000,000-sentence readiness gate with fused-AMP updates uncounted, loaded-model wording, prediction and predictive-thought certificates) and `180288c32f` (phase registry: the `fineweb` entry in `test/fixtures/phase-artifacts.json`, five certificates in `PHASE_CERTIFICATES`, an absent artifact skips and never passes; the env-var runbook retired). Tiny-run nulls, incomplete reverse programs and the 5/56 wording failures are preserved as diagnostics; old context-free and assertion-thought scores are invalid controls. No qualifying checkpoint exists; the five certificates, the three-seed equal-update causal gate and 6.5's six learning gates (no harness yet) moved to item 0; 6.2's MM_math_chain measurement was already there.

- **Item 3.5 iteration 1, lines 3–5 and the meaning/form swap:** landing `f306fc28aa` (WikiOracle `319a793`), reviewed sources `65451694e` and `01effdcaf9`, Claude LAND through Alec; scoped gates pass, the fixed-budget lesson stop and both failed modal corpus forms are carried to item 0 without rescue. The required corpus finding and parent Ethics edit are published.

- **Item 3.5 step 2c and step-2 sweep:** reviewed source `fcaff2d19`, frozen sweep repair `9f5500870`; Claude LAND through Alec, one retained full sweep and passing scoped repairs, no production change after review.

- **Item 3.5 step 2 identity threading:** landing `148953f9b` (WikiOracle `175c101`), reviewed source `65ab7fbe9`, Claude LAND through Alec; 1,627 scoped passes and twelve existing skips.

- **Item 3.5 step 2b — storage-only round:** landing `39272e7a6` (WikiOracle `ade0399`), reviewed source `6a36b7cee`; 1,144 scoped passes, nine existing skips, Claude LAND, every failure and port retained. Identity, derivation removal and 2c remain open.

- **Item 3.5 R10 — named token content before the close:** landing `e243d7bdf` (WikiOracle `2ffda78`), reviewed source `23a41540f`; exact authorized fixture correction, 111 scoped passes and four existing prerequisite skips, Claude verdict publish.

- **Item 3.5 line 1 — grammar-only reference sites and reconstruction reads the bind:** landing `a514a71e0` (WikiOracle `b49960a`), reviewed source `3a9db635e`; Claude verdict land, all scoped gates pass, R10 follows separately.

- **Item 3.5 step 1b — native binding, MTM and participant menus:** reviewed source `cc3672cfb`; all requested gates pass, seven prerequisite skips and two nonblocking root-position follow-ups declared, no learning claim.

- **Item 4 Step 0 — carried invariants and gates:** `bfc834d79`; all-level XOR/math comparison, paired reconstruction, uncapped prime/content-neutrality certificate, mention-ranking credit, one full sweep and scoped empty-menu repair.

- **Item 4 — expectation mechanism:** `fdeef42b8` on `codex/item4-repair`; approved defaults, enabled parity measurement, whole §14 completion, one full sweep and scoped repairs. Failed learning and carried Item 0/5.5 gates remain open.

- **Item 5 — forgetting:** reviewed source `b1a19c5b8`, with the approved dead-branch cleanup; LTM surprise retention, live boundary/checkpoint coverage and conservative concept reclamation. The sweep carries item 0's paired zero-cost counterexample; the link failure was environmental.

- **Item 6 — closing mechanism:** reviewed candidate `a3ec07fd0`/`b41cf3b5b`, consolidated on main: shared bounded inverse and candidate type families; forward discard with the search exemption; positional operands and orders from stamps; paired stages 1–2; reference at the close with fixed mint/bind operators and revision. **No learning claim.** At that landing, VP vectors by temporal difference, order-2 kinds and stage 4e (constituents) were not built; §15's sequence carries alongside 5.5 and at item 0, with 4e first.


- Item 6 positional operands and paired singleton warm-up: `51f069c30`, pushed on main, WikiOracle `f4d3139`. The singleton gate passes at 20/24; balanced identity remains unresolved (0/146 bindings, 0/20 row pairs), with no learning claim. Its reference-close follow-up is measured separately and accepted as the closing mechanism; learning remains open.
- Item 6 gate-off measurement and search-inverse exemption: `f1dbd12a2` (including `c647e1506`), pushed on main, WikiOracle `27a60da`; accepted. No learning claim; the fresh corrected-objective warm-up and its singleton gate are measured separately.
- Item 6 part-2 M/N/O/I preparation and first eight-branch native experiment: `af634ed35`; reviewed, no learning claim. R/S/T and all failed learning gates remain open.
- Item 6 R2 inverse-domain repair: `348fb86c7` + `90ef40a42`, pushed on main, WikiOracle `066fb5d`; accepted. Reconstruction certificates and observer repaired; no learning claim, and part 2's lessons remain open.
- Item 6 corrected-objective warm-up: accepted `345734c38`…`d5d300aed` is pushed on main, with WikiOracle bump `7bf96de`; accepted as a measurement, with no learning claim. The observation-only singleton gate was 8/24; paired training is measured separately in part 2.
- Item 6 R/S/T mechanism and matched measurement: `e38a19fbd`, WikiOracle `64a06d3`; reviewed, no learning claim. All failed learning gates remain open.
- Item 6 part-2 corpus and measurement preparation: `06f6e2b88`; M/N/O/I and the first experiment are accepted as preparation, with no learning claim.

- **Item 6 H/I follow-up** (`9cc5f6e327`): unavailable reconstruction coverage is zero; I’s empty-inverse and remaining orientation causes are bisected, with learned order still open.

- **Item 6 part 1** (`324b84d67f`): shared stored/live inverse, dropped-support keep charge, cache recovery, captured-input isolation, family-order retirement and residual/greedy components; part 2 and measured failures stay open.

- **Item 6.1** (`4d9d9ca8a`): support masks, one read per placement, eight-space STM with witnesses, derivation reconstruction and the order lesson; curriculum harness with identity/permanence and word identification passing; learning gates at item 0.
- **Item 6.2 closed** (initial mechanism `e43638a`; Alec accepted the closing October 9): mechanism complete, decomposition demonstrated; learning deferred to item 0 under the four protocol corrections—one-step problems first, worked steps as intermediate answers, binding a found candidate, and budget in steps with each query’s record charge bounded separately.

- **Item 6.5 mechanism** (Alec accepted October 7; one landing from `f4a68404e`): definedness, native identity/change columns and global bind/mint; learning gates remain pending the million-sentence checkpoint.
- `f4a68404e` **Operators update, final-b** (Alec accepted October 7): addressed rows, bipolar meanings, centroids, priming, catalogue and two-lane repair; all standing counts 10/10, R/E zero, green sweep.
- `cce3a4f7b` Operators round 3a: identity by construction.
- `3de37eef5` Operators round 2: accepted credit repair.
- `73cd7b71b` Operators round 1 accepted and pushed, with WikiOracle bumped at `c9670b5`.
- `42daf96f` Item 6.8 accepted: one attention, two spaces and one index; tag `6.8-landing`; class **7/10**, reconstruction **9/10**, joint **6/10**, MM_xor **10/10**, sum **10/10**, sweep **4,917 passed / 0 failed**, zero ownership conflicts; carried work: [FutureWork](FutureWork.md#left-unbuilt-by-the-one-attention-2026-10-04).
- `bfe6a0d7` Item 6.9: the baseline of the composition mechanism, one reconstruction/output decoder and one writer per weight; accepted class MSE .11475 and reconstruction 0/4 remain historical, with zero ownership conflicts.
- `9810fc7` Item 7: two truths, indexed definitions, ended clause state and shared row-free predicates; accepted after all three fixture ports and required checks.
- `6906727` Item 7.5: one-operation exploit/explore derivations trained at each sentence closing, reduction pressure/deadlines and closing-gradient reporting; accepted with the unchanged depth-three campaign red.
- `8bc710a` Item 9b: shared fields, parallel-first context, association-first interpretation, fixed capacities and corrected occurrence/time objectives; Claude accepted the source-matched 4,948-case sweep.
- `8bc710a` Item 9 evaluation machinery: verified FineWeb exposure and maturity-gated wording/prediction/predictive-thought checks; empirical acceptance remains open above.

- `0e70001` Item 9 parity landing: sentence-owned reconstruction candidates, final saved roots, first-sight bank checks without scalar host reads, and the historical-answer warning; expectation-learning gates remain open.

- `99207a3` Item 10 decided: dropped; normalized means deleted, native XOR and forward-evidence reconstruction pass unseeded, with mode exclusions asserted.
- `99207a3` Item 11c residue: accepted three-update refine-before-raise patience, reset by strict improvement and cleared by pure/unknown reads.
- `d5f2e1a` Item 11c: native perception, located XOR, per-pole witnessing, nonzero focus and grammatical reference across orders.
- `15c9bde` Item 11b extent correction: containment at the subject extent, observed raw spans and the restored positive-only `MM_sparse_concept` smoke.
- `aa235d2` Item 11b accepted-review corrections: recurrent parts, containment over canonical ids, and order-0 fields bound per turn with persistent concept identity.
- `db73581` Item 11b: native membership definitions, located fused parts and pervading wholes, with alternatives preserved through refinement; normalization residue closed ([design](Architecture.md#membership-read-and-extent-truth-corners)).

- `4b097dc` Item 11a learns WholeSpace properties over byte primitives, retains grounded extent evidence and passes six unseeded native XOR runs ([receipt](DevelopmentHistory.md#primitive-properties-and-grounded-extents-2026-09-23)).
- `e917d9c` Item 11 review corrections carry paired evidence through scoped dual folds, symbols, thought and checkpoints; calibrated background rejection and the full receipt pass, learned XOR remains null for item 11a ([receipt](DevelopmentHistory.md#paired-evidence-review-corrections-september-23)).
- `eb45ae4` Item 11 lands the union default, optional conjunction pass and context-backed provisional rows; unseeded XOR learning remains null ([receipt](DevelopmentHistory.md#concept-parts-and-provisional-rows-september-22)).
- `d4dc385` Item 10 (the first of that number) records the reconstruction baseline and packed/single parity gap, fixes MentalModel compaction overflow, and removes selected passing seeds (taken early from item 2; [receipt](DevelopmentHistory.md#reconstruction-baseline-and-seed-audit-september-21)).
- `f8aa23c` Item 11 completes generation catalogue ownership, scoped output dispatch, checkpoint/Adam migration and supervised gradient boundaries ([receipt](DevelopmentHistory.md#generation-ownership-september-21)).
- `afdcdfa` Item 12 completes the expectation review corrections: role-normalized retention surprise, indexed pairs, reading purity, bounded cache recovery and gradient norm ratios ([receipt](DevelopmentHistory.md#retention-score-and-pair-lookup-review-corrections-september-21)).
- `7d7dc4f` Item 2 implements negative-image expectation and residual credit; measured joint/useful-query learning remains open under item 9 ([design](ExpectationRetention.md), [receipt](DevelopmentHistory.md#negative-image-expectation-september-21)).
- `6bf211a` Item 1d restores rotation-owned concept codes and completes the 1b/1c review corrections ([GradientFlow](GradientFlow.md), [receipt](DevelopmentHistory.md#rotation-owned-dictionary-review-corrections-september-21)).
- `101dc22` Item 1c complete in mechanism: checked subsystem effects, indexed cued retrieval and retained frames; compound recovery remains unproven ([AccessibleMind](AccessibleMind.md), [receipt](DevelopmentHistory.md#accessible-mind-effects-september-20)).
- `7c2fa5a` Item 1b complete: objective-local state gradients and measured shared operator/codebook credit ([GradientFlow](GradientFlow.md), [receipt](DevelopmentHistory.md#gradient-factorization-september-20)).
- `6a6bb21` Item 1 complete: grammar-owned wording on real parsed text, held-out complete wordings/nouns, converse/sense controls and full-vocabulary generation/recomposition ([evidence and receipt](DevelopmentHistory.md#working-grammar-wording-gate-september-20)).
- `d1d8b5b` Ordered compose-MLP inputs and production generate wiring (item 1 increment; the wording gate remains open, [SelectedMeaning](SelectedMeaning.md), [Testing](DevelopmentHistory.md#grammar-owned-wording-architecture-september-20)).
- `02db7ef` Review corrections: removed the standalone codec and legacy policy; completed chooser context, bounded memory and nested-controller mechanisms ([SelectedMeaning](SelectedMeaning.md), [Testing](Testing.md)).
- `288b56b` Initial controller replacement; review reopened item 1's wording gate ([current design and limits](SelectedMeaning.md), [original receipt and review corrections](DevelopmentHistory.md#selected-meaning-and-one-controller-september-20), [test dispositions](DevelopmentHistory.md#frame-kernel-test-migration)).
- `60b1497` Forward owns anchored lexical forms by WORD row; capture reads retained provenance ([Testing](DevelopmentHistory.md#forward-owned-lexical-forms-september-19); item 1 increment).
- `a70c77d` Direct `concept`-unary selected meaning (`what(quantize(x))`)
  preserves its live leaf into the normal controller; description operands
  still require owned occurrences. Ordinary pytest failures now complete their
  selected receipt while resource/protocol failures remain bounded/fail-fast
  ([Testing](DevelopmentHistory.md#complete-diagnostic-coverage-september-19)).
- `943a64f` Lexical converse forms: `whole` vs `part` provenance frozen at
  program capture ([Testing](DevelopmentHistory.md#forward-owned-lexical-forms-september-19)
  records the later forward-ownership correction).
- `fb5611c` Normal thought resolution grammar-owned; `resolveAnswer()` no
  longer calls `answer_query()` from a raw prompt (item 1 increment).
- `ea92851` **Item 0 complete:** model-declared `<thought>` allow-list between
  `<compose>` and `<generate>`; `<Queries>`, the `is`-aliases and
  `query="false"` retired.
  [Receipts](DevelopmentHistory.md#explicit-thought-catalogue-september-19).
- `573fe0c` Bounded tests fast: no background QoS, `cpu−4` one-thread pool,
  28 GiB aggregate cap; 4,649 cases in 197 s (was 4,165 s serial).
- `6a34bbf` Sentence expectations retained as `estimate` occurrences without
  LTM leakage ([ExpectationRetention](ExpectationRetention.md); item 2 foundation).
- `c3918a2` `9dc5c58` Typed `ThoughtResult` and nested evidence persist through
  history checkpoints (sidecar v3; [ThoughtHistory](ThoughtHistory.md)).
- `e8b2f43` Catalogue refinement preserves mode, polarity, bindings, scope.
- `b705a48` `69233d1` Selected truth and `arma` results become the row's answer
  seed; `arma` emits `[3, D]`, never a fact.
- `04b34a3` Normal What runs on the selected thought trace.
- `4518db0` Structural and thought operators unified (predecessor of `ea92851`).
- `7923845` Shared selected-query work meter ([QueryWork](QueryWork.md)).
- `3ab3c4b` Nested grammatical occurrences retained ([NestedRetention](NestedRetention.md)).
- `3a5703e` Checked query execution phases ([QueryPhases](QueryPhases.md)).
- `985b594` Replayable ordinary thought history ([ThoughtHistory](ThoughtHistory.md)).
- `3ffb465` Bounded runner foundations, 512 MiB VQ tile, MPS routing receipt.
- Expectation review: implemented and validated on September 16.
- Xcode licence accepted; trust term sign `|trust|` (Alec, September 18).

**Update 2026-11-08 (§29.2/§30/§31, argument concepts):** Alec's representation correction
(§29.2, recorded by Codex): formal arguments are learned concept vectors in the existing
hierarchy, pre-built one order above word concepts, directions from members' meanings,
bidirectional conditioning, no parts-of-speech special case beyond construction and grammar
use. Codex built it (§30; source
bcf39ff2..., 1005 files; sweep 7475/26/323/1 xpassed, red, every failure attributed).
Claude's design review §31: the representation reads grammatical function from meaning
(Alec's bet, the participation-beats-resemblance certificate is the test); finding 9 the
meet-ratio agreement caps a class radius at its members' spread (0.69 for two orthogonal
members, the 0.75 gate unreachable; D13 agreement = centroid stability); finding 10 every
member of every class is conditioned every sentence and a class collapses (cosine 1 in 20
boundaries; D14 condition the boundary's members only, refresh once); finding 11 word
concepts stay order 0 (fixture port); observations on evidence-less membership, 96 seats,
thought proofs over grammatical classes. Receipt review §31.1: not accepted (part_edges
double-counts relation-backed members, the centroid schedule, the seeded XOR learning
certificate at .234 with conditioning the prime suspect, 22 fixture/inventory ports;
finding 12 zero operation trust on form-only charts, D16 neutral factor). Codex text §31.2
(return §32); green there freezes the round; then the §5
review before the push.

**Update 2026-11-08 (§31.3/§31.4, Alec's answers):** Alec accepted the hierarchical training
regime; his answers re-design the read and the measures: D17 the class read is coverage of
the class vector by the word (meet ratio; low magnitude = lenient, so a dominant class cannot
starve a new part of speech); D13 revised: definiteness = separability (fit-weighted fraction
of members nearer their class than any competitor outside the 15-degree contiguity angle) x
evidence (1-exp(-views/trustViews)), the stability D13 withdrawn, the meet-ratio agreement
kept as tightness; finding 10 reframed (word-concept contraction toward the class centroid is
the membership degree; D14 keeps it following the word's own participations); D15 members
weigh their fit in the centroid; D16 kept; word-order separation (form order vs referent
order) recorded as a later item; thought over grammatical classes allowed and recorded
(Orders.md new subsection). Codex text §31.4 supersedes §31.2 (return §32).

**Update 2026-11-08 (§32/§33, argument-concepts-ii reviewed):** Codex built D13 revised, D14-D17
and the part_edges fix (§32;
source 037afb6b..., 1005 files; sweep 7505/4/323/1 xpassed). §33: not accepted; two ports
(a source-inspection literal in test_serial_concepts; Codex's own §30 order test under the
old read) and finding 13: the grammar-XOR learning certificates regress (the seeded one at
.2355 again; the unseeded test_xor_class_accuracy newly, two answers wrong); prime suspect
the fit carrier added unweighted to the chooser's scores (co-occurring words fit any young
class by ~0.78 once placed), not established; a controlled attribution panel (§32 source,
carrier silenced, §30 baseline; 3 inits each, ~40 min) is proposed for Alec to authorize;
the seeded certificate's standing is his. Codex text waits on those two answers.

**Update 2026-11-08 (§33.1/§33.2, the XOR panel):** Alec: run the panel here; move the seeded
certificate behind RUN_TRAINED. Panel:
nine unseeded XOR_grammar trainings: §32 as built 2/3 pass, carrier silenced 0/3, §30
baseline 2/3. Finding 13 closed (the carrier helps; no regression from the round; D17 stands).
Finding 14 opened (FutureWork): the grammar XOR training does not converge in about a third
of initializations (the loss oscillates between ~0.05 and 0.25 for all 400 epochs; momentary
dips near zero do not hold), on §30's source too; the unseeded
certificate's "arbitrary initialization" bar is a coin with a one-in-three tail. Assumption for
Codex text §33.2: it moves behind RUN_TRAINED too (trained('xor_grammar')), the seeded one as
trained('mm_xor_grammar'); two ports; return §34; green there
freezes the round. Alec's aside: tiny trained models (few words) as registered artifacts for
the trained tier (lessons item).

**Update 2026-11-08 (§34/§35, argument-concepts-iii accepted):** Codex's return (source f770cd13..., 1005 files) changed
only the two ported tests, the two trained markers and the phase registry; make test 7509/325/1
xpassed, exit 0. Accepted (§35): f770cd13... is the argument-concepts round's frozen source and
item 2.3's complete code (rounds 1, 2a, 2b-in-§30, argument concepts). Next: the §5 review with
code in hand before the push, on Alec's word (carry D8-D17, Q2(b), the word-order convention and
FutureWork item, finding 14, the lesson measurements, tiny trained artifacts). Then item 2.0
round 1 (handoff task 4).

**Update 2026-11-08 (§36, the §5 review before the push):** §5's seven decisions re-read
against f770cd13... with the code in hand: all stand as decided and amended (D1-D17, Q1,
Q2(b)); two wording changes recorded (the source trust scales the stored pair and the
learned factors are a minimum at the read, Alec's §9.3 form; the kept-use count is
telemetry, the class views being the count factor). §36.2 lists seven items for Alec,
each with a recommendation: confirm D13 revised/D15/D16/D17 as built; the m_o change; the
composition form; the two-gate coarse order; gate data defaults (push as frozen, change in
the lessons item, where a `<minimum>` word count for the interpret gate also belongs); the
word-orders item after 2.0; finding 14 with item 0's learning findings. §36.3 is the Codex
landing text ("Land accepted item 2.3 implementation", record the landing, push main);
Alec's word: Codex commits and pushes on completion. Receipts stay local (doc/benchmarks is
ignored). Next: item 2.0 round 1 (handoff task 4) after the landing.

Landed as 59cd46607744350dc905aca2effe3df12aaabfaa, 2026-11-08

**Update 2026-11-08 (item 2.3 landed; item 2.0 round 1 text):** item 2.3 is landed and
verified (BasicModel 59cd466077 + 87e74ff435 on main, pushed; tree clean; source hash
f770cd13... recomputed; no model identifier in the messages); WikiOracle's submodule bump
d8310f6 "Land accepted BasicModel item 2.3" is committed, push pending (Alec). Alec:
"issue text for both you and codex to begin 2.0". Written: the hand-off (state, text for Claude,
text for Codex), the rounds and the round-1 Codex text (bin/Memory.py
with STMChunks, ShortTermMemory, MediumTermMemory, LTM0 and LTM1 moved and the three old
modules deleted; thought ideas stamped and found by .when;
ReferenceClose.commit's substitution and word_occurrences retired, §54 finding 3;
no row for a SHIFT at the source, §58's note; documents; fast checks; return as §11). Round 2's
measurements conflict with the fast-checks rule: put to Alec before its text.

**Update 2026-11-08 (round 2 under the fast-checks rule):** Alec: "Fast checks only." Round 2
builds its repairs (fail loud in the reverse and `runBatch`, the derivation inverse given the
forward's action context, one WholeSpace, the supplied `yes` admission, the sentence-82 capture
rule) with direct cases, the targeted files and one `make test` at its freeze; item 71's runs and
the item 60 metrics move to the end of item 2.0, taken on round 2's frozen source and on the
finished tree. The round-2 text follows the round-1 review.

**Update 2026-11-08 (§11/§11.5, round 1 reviewed and accepted):** Codex's round-1 return
is reviewed in §11.5 and accepted: the
frozen source `2d2abfe10878…` (1004 files) equals the live tree, the before-source is the landed
`f770cd13…`; `bin/Memory.py` holds STMChunks (a namespace class), ShortTermMemory, MediumTermMemory,
LTM0 and LTM1 with the three old modules deleted and every moved function unchanged by AST; thought
ideas carry their thought step's `.when` (the former address digits, offset 2, disjoint from model
time) with an empty `.where`; the one-leaf substitution and `word_occurrences` are retired; a SHIFT
gets an empty sentinel row and no row-0 copy; one `make test` 7521 passed / 325 skipped / 1 xpassed
(= 7509 + 11 new + 1 link case); Claude's targeted run 2089 passed / 55 skipped. The stateless
`MemoryStore` base, outside the §10 text, is Alec's direction (confirmed 2026-11-08: a base class to
consolidate code; `Memory` is the module's name). Landing text §11.6 (Codex commits and pushes on
Alec's word). Next: round 2's text (repairs under fast checks, §10.2) after the landing.

**Update 2026-11-08 (§11.7, landing and round 2 in one text):** Alec: "add the round two text to
this prompt, otherwise we take too many iterations to finish 2.0". One Codex
text (§11.7): steps 1-3 land round 1 (2d2abfe10878…, push), then round 2 starts from the landed main: fail
loud in `_reverse_body` and `runBatch` and the `_inputShape` repair; the derivation inverse and the
walk recomposition given the forward's action context; one WholeSpace with the checkpoint migration
and the persisted-address certificate; the supplied `yes` admission's three-stage report (home: this
round, unless Alec moves it to item 0); the sentence-82 capture rule at `ReferenceClose.commit`'s
`mtm.bind`; the ≥1e8 item recorded empty. Fast checks only; the measurements wait for the end of 2.0
(§10.2). Return as §12.

**Update 2026-11-08 (yes admission's home):** Alec: "Yes, round 2". The supplied `yes` admission is
built in 2.0 round 2 (§11.7 item 7), not item 0; the §11.7 text has no open assumption.

Landed as 6becc3dc47c88d787406ff94f1a4ba6acbf95eec, 2026-11-08
- 2026-11-08 (evening): item 2.0, the memory amendment: the derivation tree (operator occurrences in STM0, ordered binary branches in STM1) is the pinned sentence's working structure and collapses at the kept close; LTM holds compacted ideas and relations only; no new rows "here or anywhere, without a lot of justification" (Alec). Open for Alec: whether `LTM1.operator`/`grammar_operation` stay as the collapse's residue.
- 2026-11-08 (evening): item 2.0, the memory amendment: DECIDED (Alec: "keep them until the round that moves operation authority onto the operator concepts' trust") `LTM1.operator`/`grammar_operation` stay as the collapse's residue until that round, which retires them; no earlier round touches them.
- 2026-11-08 (evening): item 2.0 round 2 at an interim return (items 4–9 built, no freeze, no sweep); §13 withdrew the item-5 routing blocker; continuation Codex text given (item 5 free generation checks a candidate in its own object basis and group, replay keeps the record; then the freeze; return as §14). The memory amendment is the memory round's planned work.
- 2026-11-08 (evening): item 2.0 schedule with fewer passes (Alec: "Merge 3 and 4"): rounds 3 and 4 one round; measurements inside round 6's freeze; landings folded into the next text; reviews accept with warts unless a gate fails; three Claude texts remain after round 2's return (rounds block).
- 2026-11-09: item 2.0 round 2 REVIEWED and ACCEPTED: frozen source `8cde24380d44…` (1008 files) equals the live tree; before = round 1's `2d2abfe10878…`; one sweep 7551/325/1 with 1 failure (the `test_operators_round1` walk-teacher fixture lacks the new `PrimedSymbols.verb_groups` field; a one-line fixture port at the landing, not a producer `getattr`); gates passed; six certificates and five targeted selections reproduced; six warts recorded (a sloppy `_checkpoint_model` port, two dead WholeSpace flags, the band-recognition heuristic in the address migration, a host loop in `walk_bank`, the traceback-walking capture). To confirm with Alec: land with the two one-line test ports and a targeted rerun, or a second full sweep.
- 2026-11-09: item 2.0 round 2 landing CONFIRMED (Alec: "Fine with landing"): the two one-line test ports and a targeted rerun, no second sweep; landing folded into the next Codex text with the merged round 3/4.
- 2026-11-09: item 2.0, one Codex text GIVEN: land round 2 (two test ports, two-file rerun, commit, record, push) and build the merged round 3/4 (`separateChooser` false with the two-sided projection per unit and its receipt; the sides of §2 as §8 amends them; generate under compose's proposal and `score_function` credit, `walk_policy` retired; verb and modal teachers retired, the assigner kept on the recorded verb-phrase meaning, the re-read at `PrependedSymbols.symbolize`, `training.actionRateRatio`; the XOR gate once; output typing as generate-policy features from the `argout` faces; §54 finding 5's clamp velocity repaired); fast checks only; return as §15. Three texts remain two: land 3/4 + build 5; land 5 + build 6 with the measurements.

Landed as acbe466e871df474f864bfd403e11c0921441529, 2026-11-09
- 2026-11-09 (Claude): round 3/4 §15 return REVIEWED: round 2 landed (acbe466e87 + e4b72d6989, pushed); frozen 40676b500b3a… equals the live tree; gates green; sweep 7513/325/1 + 49 failed = 48 test ports (seven causes, each verified on a scratch copy) + one defect (cross-sentence graph carry at the single-path sentence boundary, packed streams, nondeterministic) + one latent width defect (dense-synthesis placeholder 5 vs 9) + the degraded separateChooser-true path. NOT landable as frozen; one repair pass then the landing. Open for Alec: delete the separate path (recommended) or repair it; repair+sweep+land-on-green in one Codex text (recommended) or review the repair first.
- 2026-11-09 (Alec): the `separateChooser` true path stays ("keep it, it's an option, not a legacy path"), its routing repaired (owner and side as two term attributes); one Codex text repairs, sweeps once and lands round 3/4 on green. GIVEN (the carry repair with its structural certificate; one walk-feature width; owners for the option, sides for the default; the 48 ports; the one-line warts; documents; sweep then landing in the §11.7 form, "Land accepted item 2.0 round 3/4"); return = §16 (landing record or the red cases). Then: build round 5 (the memory amendment, basic percepts, §54 finding 2); then land 5 + build 6 with the measurements.
- 2026-11-09 (Claude): round 3/4 repair return (§16) REVIEWED: production repair accepted (carry defect closed by `SentenceBoundary.finish` + retained traversals, 8/8 solo runs; one walk-feature width; owner and side per term, the option's routing restored; the one-line warts); fourteen direct certificates equal; gates green; sweep 7553/324/1 + 12 failed, all fixture ports (ten toy languages without `parts_of_speech`, one 5-lane fixture, the pullback-count list). Not landed. Alec (2026-11-09) on the "Thinking is Optional" verification: the architecture must present dual process theory (System 1 and 2, animal vs human); MTM stays grounded in object files (references stored, origin unchanged); MTM on basic percepts is a development stage for the next-order concept; text for Codex, to be added to Architecture.md and Philosophy.md at the round-5 build. Open for Alec: land with the 12 ports and a six-file rerun (recommended) or one more sweep; then the land-3/4 + build-5 text.
- 2026-11-09 (Alec: "good recomendations"): round 3/4 lands with the twelve fixture ports and a seven-file rerun, no second sweep; the operator seats declare identity and execution interface at construction and learn direction, confidence and participations. ONE Codex text GIVEN: steps 1–3 land round 3/4 ("Land accepted item 2.0 round 3/4", record line, push); items 4–10 build round 5 (the one `link` over references with computed values and promotion of paths; `memory.verb_state`/`memory.modal_state` as relative errors over the state of affairs; shared STM0/STM1 strata, the pinned sentence's tree collapsing at the close, operator concept seats, authority onto their trust with `grammar_operation` becoming the seat reference; basic percepts with MTM as their writer, supplied forms declared at admission; the elided role's content; the dual-process and memory documents); return = §17. Then: land 5 + build round 6 with the measurements.

Landed as b7d0914ab33987ce8f786f400298bc4bc6ee2075, 2026-11-09
- 2026-11-10 (Alec): episodic memory "can happen now as long as we only store a few percepts (let's set that as a model parameter). It should come from the open-attention round." Recorded, with Claude's proposal (an entry beside each `LTM0` row, `episodicPercepts` percepts per entry, read from `FamiliarTier.open_sentence`), in [FutureWork §7](FutureWork.md#7-episodic-memory-a-few-active-indices-beside-each-row); `.when` is model time in every store, with the drift remedy "we will need to reset the time when we reset the epoch", in [FutureWork](FutureWork.md#when-is-model-time-in-every-store-alec-2026-11-10). Placed later the same day as round 7 of item 2.0 (next line). That round also moves the remaining `.when` stamps to model time with the epoch reset (Alec, same day; next lines).
- 2026-11-10 (Alec, later): "lets schedule episodic memory as a round inside 2.0": round 7 of item 2.0. Rounds 5 and 6 (return §20.21 to §20.34, freeze `eb81e90f4d7f`) reviewed and accepted; they land as the first three steps of round 7's text.
- 2026-11-10 (Alec, question tool): round 7 decided. The stamps still per document move to model time in the same round, one tick per sentence ("One sentence") and per thought step, one clock for thoughts and the MTM pool ("One clock"), both clocks put back at each epoch reset; entries beside `LTM0` rows ("in particular, beside LTM0"), `episodicPercepts` 4; review rides on the thought loop's time-only recall.
- 2026-11-10: item 2.0 rounds 5 and 6 landed as 16a6a79b81fb1cc21d116d71f1f6eea389553703; accepted, frozen source eb81e90f4d7f, focused validation 875 passed / 29 expected skips. Round 7 follows from this landing.
- 2026-11-11: item 2.0 round 7 stopped for review at source `6bf63e0922ce`: the direct clock certificate's native convergence prerequisite reached MSE 0.24558 against <0.20 after 200 epochs. The clock, epoch reset, relative readers and existing-LTM0 episode/review implementation are in the working tree; 132 targeted cases passed, followed by 264 passed / 6 skipped. The return records the raw word-expectation reader repair, its unfinished validation, and the relation-only closing decision. The eight fits, four lesson arms and final sweep remain pending; no round-7 commit or push.
- 2026-11-11 (Claude): round 7's stopped return reviewed: the landing of rounds 5 and 6 is exact; source `6bf63e0922ce` and the two fast runs (132; 264 passed, 6 expected skips) reproduce; built as §21.1 asks; the convergence stop stands (that draw did not learn XOR: MSE .2456 against .25 for answering one half), its cause open. Three findings by reading: thought rows copy the sentence's entry, `ConceptualSpace.End` keeps the attended field across batches, and one thought turn is keyed by both time and address. Open for Alec: the entry of a relation-only closing.
- 2026-11-11 (Alec, question tool): a relation-only closing gets "An LTM0 anchor row" for its entry.

- 2026-11-11: item 2.0 round 7 continuation implements the input-only episode writer, the decided relation-only LTM0 `episode` anchor, `End` cleanup, and document-ordinal keys in both thought branches and document covers. The recorded convergence draw differs from landed main despite identical initial parameters; the conceptual mixer now reads elapsed-time differences. No round-7 commit or push.
- 2026-11-11 (Claude): Codex's continuation (§23, in progress at source `7f26c4043fc8`) reviewed: the convergence regression came from the conceptual mixer reading the absolute band and is repaired (`ConceptualSpace._bind_read`, `RowTime.difference`; 8 of 8 fits on each side); §22.4's three findings and the anchors are built as §22.5 asks. Open: one lesson arm red at epoch 2 ("recorded concept order disagrees with its parts").
- 2026-11-11: round 7 resumed after restart on verified `7f26c4043fc8`. The saved lesson draw passes on landed main but exposes a state-link order mismatch on round 7; the trial and commit now read the retained occurrence's actual order. Targeted validation is 272 passed / 6 expected skips after two clock-fixture ports. No round-7 commit or push.
- 2026-11-11: round 7's lesson-order and thought-view repairs pass the recorded 102-sentence replay. Final source `da78077358c5` completes all four lesson arms and eight 400-epoch XOR fits (8/8 numeric and grammar gates); lesson answers remain 0/320, with 24 time-only recalls and 8 reinstatements. The full frozen-source sweep follows; no round-7 commit or push.
- 2026-11-11: round 7 returned for review at `fde035b1464d`. The full sweep completed 8,058 cases with all required gates passing; its 76 actionable failures are repaired and verified by whole-file/importer checks (latest 1,531 passed, 60 skipped), with the sparse-concept failure carried to item 1.5. The measured and swept source `da78077358c5` is distinct from the final repair source. No round-7 commit or push.
- 2026-11-11 (Claude, at Alec's request): documentation pass on episodic and semantic memory and on System 1 and System 2. Dated amendments and no code: [Architecture](Architecture.md#episodic-and-semantic-memory-in-the-two-systems-2026-11-11) (header, grounding bullets, epoch boundary, thought time, the two inverses, a new subsection and two status rows), Philosophy, STM, Training, Reasoning, GradientFlow, Params, and FutureWork §7 and the model-time section. Round 7's parts are marked "working tree, not landed" and need that marker changed when the round lands.
- 2026-11-11 (Claude, at Alec's request to begin the compiler work): item 1's read-only baseline: on the landed main a lesson sentence spends 70% of its training time in the generate walk (22 walks, about 615 `reverse_binary_step` calls), run as Python because the lessons run uncompiled; backward is 5%; item 1.5's original hot spots are a few percent on lesson-13. The default sweep spends 61% of its worker time outside the test bodies, and compiled against eager makes no difference to the two files timed. Proposed order, for Alec: the generate walk first (item 1), then item 1.5's port, on a separate branch (Alec: "That can be done in a separate branch"). Observed in passing: `test_output_walk.py::test_runbatch_answer_availability_never_reassigns_the_decoder` fails on the landed main.
- 2026-11-11 (Alec: "You can do the generate walk first"): item 1 starts with the generate walk, on branch `item1-generate-walk` in a second worktree (`WikiOracle/basicmodel-item1`, cut from `3c9399199d`, nothing committed). Measured first: why a sentence makes 22 walks, and that 56% of their menus are exact repeats. First step built there: `Generative.MenuStore` keeps menus between the walks of a batch; the same ten seeded sentences give identical parameters and costs after every batch and take 23% less time (57.4 s against 44.2 s, one thread). Its regression run and a sweep are outstanding.
- 2026-11-11 (Alec: "Maybe they could all share the same pair"): of 88 pairs measured, 48 have no legal departure (the explore walk is the greedy walk again), 41 greedy walks and 31 whole pairs repeat an earlier one exactly; the twin never shares the presented reader's idea. Built on the branch: no second walk without a departure. The menu store now lives for one batch only and a named reading sets it aside, after the grammar gate's spy case failed on the first build. On the final source (`88dc4d653af4`): identical seeded updates, 27% less time (54.1 s against 39.7 s), and the fifteen walk-related test files fail only the two cases that fail on main. Two cases are red on the landed main itself (`test_output_walk.py`, `test_concept_output.py`). The design options for sharing pairs further are Alec's.
- 2026-11-11 (Alec: "I'm pretty sure that we only need one grammatical pair across all compose, thinking, and generation"): what stops it was measured on 20 lesson sentences. Nine of a sentence's eleven read-out pairs only price a compose or thought trial, each as a whole pair. The twin reads the presented reader's roles (40 of 40) and differs by one weight, `question_conditioners`, which trains in the twin and stays exactly zero in the presented reader (a finding for Codex). Reconstruction unfolds `record.end_slots`, the answer reader the sentence's meaning: the same tensor in 35 of 40 trials, then moved by the answer attention's read. 39 of 40 thought trials are priced by reading the kept composition again with fresh draws, and in 4 of 19 sentences the explore thought was kept on a difference that came only from that draw (a second finding). The presented output starts from a third idea in 18 of 20. Arrangements: one pair for each stage, 5 to 9 walks a sentence, or one departure a sentence, 2 to 4; today 22. The choice is Alec's; nothing built.
- 2026-11-11 (Alec, question tool): "One pair per stage"; the twin: "Retire it for walk readers" (`XOR_grammar` keeps its twin); and a composition is priced through its thought: "we want both reconstruction and output to influence a single understanding; so if I understand correctly, we need to price it through thought". Together: each composition priced by one greedy reconstruction walk and by its greedy thought and one greedy output walk; the kept composition's reconstruction read-out, thought and output read-out each explore once; one reader where `answerSynthesis` is true. 4 to 7 walks and 2 or 3 thoughts a sentence, against 22 and 2. Open for the specification: the thought before the commit (round 7 is changing that code), the question conditioner and attention read, the reader's training exposure, the gates, and who builds. Nothing built; no Codex text yet.
- 2026-11-11 (Alec): "Codex, after round 7" builds it. His concern that "XOR_grammar's twin" is a workaround is borne out by the record (added 2026-10-06 for an affine reader that could not read one composition; that cause was withdrawn on 2026-10-07; the twin stayed), and he decided "Retire it everywhere". Specified (one pair for each stage, one reader): a composition priced by one greedy reconstruction walk and by its greedy closing thought and one greedy output walk; the thought before `_commit_sentence` writes; one explore each for the reconstruction read-out, the thought and the output read-out; $A$ trains from the output read-out's kept walk alone; `AnswerComparison` and the shared draw table deleted; the batch end replays. At most 7 walks an asked sentence against 22. Gate: ten unseeded `XOR_grammar` trainings without the twin, the lessons, one sweep. Two findings to diagnose in the round: the presented `question_conditioners` weight never trains, and item 2.3's finding 14. Codex text to be given after round 7 has landed. In a toy, one reader settles in 40 of 40 seeds where both compositions are readable.
- 2026-11-11 (Alec: "quick take: why wouldnt the answer train on both cmopositions?"; then "Yes, both"): the one reader now gets the twin's exposure rule (`SentenceCredit.reader_weights` with the draw): half from each composition where the departure lay in compose, the kept composition alone otherwise (another reading of the input). Its recorded cost is stated with it (the rejected composition's share does not decay; class 2 of 10 in round 2d, on roots that interfered), and the round's measurement reads the reader's cost on the kept composition for it. Toy corrected: every rule settles by 1,600 epochs, so the rules differ in speed; one reader on both is slower where both compositions are readable and keeps a small error on the kept one. Codex text amended (items 4 and 8).
- 2026-11-11 (Alec, debating a keep at the end of every stage: "the keeping acts like a local minimum on the error surface, whereas if we have one departure per sentence our minimum is evaluated only at final stage over all stages. It will probably be slower, but more thorough. If we wish to increase the credit, we might introduce a second explore path"; then "Yes, default one explore"): the design is rewritten for whole paths. Claude's claim that the stages do not interact was wrong: the reconstruction walk's words become the order-zero field (`OutputOccurrences.generated`, `SentenceField.publish`) that the thought reads (`_selected_thought_memory`), so the stages are a chain and a keep on $R$ alone is a keep at a local minimum. Now: one exploit path and `architecture.explorePaths` (default 1) explore paths, each with one departure drawn by `SentenceCredit.departure` over seven walks, sharing the exploit path's trunk, all compared once on $R + A$; one credit a path; `generation_pair`, `output_pair` and `thought_pair` deleted with the twin. At most 4 walks an asked sentence against 22. The lesson measurement runs with one and with two explore paths. Codex text rewritten; it replaces every earlier version.
- 2026-11-11 (Alec: "Yes, count silence"; and "Output is never a direct consequence of understanding the question; it is a consequence of the answer produced by thinking about the question"): written into the whole-path round. On data that supervises output every row has an $A$, the supplied answer where one was asked and silence otherwise, in the keep and the credit as well as the training; `_answer_trial_cost`'s `asked_only` form goes and no walk leaves the draw on a statement. The output walk's operand is what the path's thought concluded and nothing else: zero where it concluded nothing; `_resolve_answer`'s `identity` default and `_sentence_reader_error`'s read of the sentence's meaning go at a sentence close. To specify next, with item 0.5's corpora: text that supervises no output, where Alec proposes degraded input reconstruction as the question the text answers itself. Codex text amended (items 1, 2 and 9).
- 2026-11-11 (Alec, a direction for before item 0): "we will want to create an adaptive loss function over the input to teach the model before item 0, so minimally a maskRate for both forms and words. It ought to be able to drop both percepts and concepts. Ensuring a high, long-term reconstruction rate (e.g. 90%) would allow learning while at the same time creating an raising bar for prediction"; omitted "using an out-of-band signal, not corrupted, because the best model reproduces corruption on seeing corruption". Traced: nothing is hidden on the sentence path today; `maskRate` masks percept slots only on the path without the per-word loop; the all-zeros mask vector is still there (`create_ir_mask`, WHAT zeroed, WHERE and WHEN kept). Not specified; it goes with the degraded-input regime (item 0.5's corpora).
- 2026-11-11 (Alec: "ask your questions: I want to spec this before the gym"): omission is specified inside the whole-path round. His answers: the rate held is "Of everything"; restored by "Reconstruction only"; a dropped word loses "Its whole form"; "On by default, but you can override to get existing tests to pass"; built "Inside the whole-path round". Two sizes (one percept slot of a kept word; every slot of a word), zero content with place and time kept, nothing the model reads carries the omitted content, reconstruction's targets are the input before the omission, each rate $\operatorname{clip}((v-r)/(v-a), 0, m_{\max})$ from long-term accuracies on visible and omitted items; `architecture.omissionTarget` (.9), `omissionLimit` (.5), `omissionHorizon` (1,024); target 1 hides nothing and is the override for existing tests and the XOR proofs. The Codex text gains item 9 and a third lesson arm. Open (Alec, same day): "we need a way for the monologue corpus to train system 2"; the record's answer is the next sentence, not built for the thought; which round builds it is his.
- 2026-11-11 (Alec: "No, I do not think predictive processing is a good solution; reconstruction is better... we might do reconstruction as augmented by compose() and generate() without doing any thinking()"): the next-sentence judge of the thought is not to be built. What he describes is the order-zero reconstruction as built since 2026-11-10 (`SentenceField.reconstruct`: the generate walk on the composed end state, its words as the order-zero field, then `reverse`), so omission trains compose and generate through it; only the thought is outside. Recorded in the whole-path round (the input reconstruction is costed only where content was observed: Claude's choice, for Alec's yes); Codex text item 9 amended.
- 2026-11-11 (Claude): round 7's return on `fde035b1464d` is reviewed and **accepted**. Verified: the hash; seven production files since the interim review, read; after the measured freeze only `bin/RowTime.py` and 33 test files, and 20 seeded lesson sentences identical with that change reverted; 49 fast checks passed here with both stopping gates. Warts carried: optimizer-gradient peaks up in three lesson arms, answers 0 of 320, the swept source is the measured one. The landing is folded into the next text, the whole-path round's: steps 1 to 3 land round 7 and items 5 to 14 build whole paths, one reader and omission. The "not landed" markers of the documentation pass now read "accepted 2026-11-11".
- 2026-11-11 (Alec: "Please add to future work an investigation of this possibility, particularly if system 1 might learn to understand whole sentences without grammatical operations"): [FutureWork](FutureWork.md#a-fragment-or-a-sentence-understood-in-one-attentional-window-without-grammar-alec-2026-11-11-to-investigate), a new entry: what exists (a fragment closes as an idea; the open attention pass; a recurring phrase minted as one concept; expectation trained on System 2's order-zero contents), what is not known (nothing requires the window's reading and the composed one to agree), and three measurements (phrase parity; whole sentences with the compose walk withheld at test; when attention descends). It waits for the whole-path round and a corpus on which phrases recur. No mechanism proposed.
- 2026-11-11 (Claude, on Alec's "Review, please"; no return from Codex is on disk, main is at `3c9399199d`): the whole-path round's text reviewed before its hand-off, four repairs recorded. The first matters to the XOR gate: decision 7 (the output's operand is the thought's answer, empty where it concluded nothing) cannot apply to a numeric head, since `XOR_grammar.xml`'s closing thought concludes nothing in 48 of 48 sentences and its affine head reads the composed state; the head reads the path's sentence state as now, for Alec's confirmation. Also: `_answer_occupancy`'s fallback to the field's role mask goes at a sentence close; the omission draws and the accuracies' decode are stated exactly. The FutureWork entry gained Alec's existence argument and its limits. Codex text amended in items 5, 6 and 13.
- 2026-11-11 (Alec: "We have converged on a model design at this point: so the purpose of the xor test is to test that design on the xor problem, not to make a new model architecture"): the numeric-head exception proposed in the review is withdrawn. Specified: `XOR_grammar.xml` keeps its four sentences, classes and compose grammar and is posed on the design (`answerSynthesis` true, the class supplied as an answer word, concluded by the thought and said by the output walk); `test_xor_class_accuracy` becomes four exact answers; no thought, head or rule is added for the fixture. Likely red at first (lesson answers 0 of 320): reported and diagnosed. Not decided: the other XOR proofs and numeric heads elsewhere. Codex text: item 5 amended, new item 14, certificates now item 15.
- 2026-11-11 (Alec: "Yes, it stops the round. A continuous output (and continuous input) can be treated as alternate modalities. Using a linear output head on top of order 0 concepts instead of verbal reconstruction can be a model option. So with that caveat, you can keep xor close to its current form."; and, shown the measurement, "Sorry; when I said order-0, I meant no grammar (higher-order concepts are in fact required for xor)."): the answer-word form of the XOR gate in the line above is superseded. Specified: a continuous output is a modality of the design, a linear head on the concepts (higher orders included) with no generate grammar; `XOR_grammar.xml` keeps numeric classes, `answerSynthesis` false and today's `_forward_head` on each path's composed sentence state, loses the twin, and is a stopping gate (ten unseeded trainings, five or more passing; the count is Claude's, today's rate being about two in three). Measured before specifying: the order-zero field generate unfolds from the understanding is exactly the two words at amplitude 1, so no linear read of it gives XOR; the fixture's thought concludes nothing in 48 of 48 closes. Not decided: what a continuous head reads where a thought concludes; continuous input. The Codex text is final again.
- 2026-11-11 (Alec: "Please add a numeric input as future work; xor should probably be converted to use they when it exists. Probably xor should be converted to a system 1 model. That means just one concepts-to-concepts step at order 0 (Which is roughly equivalent to its previous sigma pi, I think?) And in fact, we probably want to have only a BasicModel class; I forget if we have other models being tested?"): [FutureWork](FutureWork.md#numeric-input-as-a-modality-xor-as-a-system-1-model-and-one-model-class-alec-2026-11-11-to-specify), a new entry with what exists: a `numeric` data type (dense slab; `model.xml`, `mnist.xml`) that no XOR configuration uses; an XOR without grammar already in `XOR_exact.xml` and `test_grounded_xor.py` (conjunctions over order-zero memberships, then one learned sigma row of order 1; character input); two classes only, `BaseModel` and `BasicModel`, the factory building `BasicModel` and nine test files constructing a bare `BaseModel()`. To specify: numeric input as a modality, XOR on it as one concepts-to-concepts step with the linear head, one class. The whole-path round is not changed.

Landed as 984a75b20e88d5b052011adb31ccd90d28234519, 2026-11-11
- 2026-11-11 (Alec: "This is actually current work: it will clarify the architecture to store concepts instead of percepts. That will allow us to do animal cognition (system 1 reconstruction) without calling compose and generate."; "So that would be an immediate change."): the episodic entry is to hold order-zero concepts of the open attention pass, not basic percepts. Item 16 of the whole-path round's Codex text (built first after round 7 lands): `EpisodicMemory.stage` selects from the open pass's concept evidence; `SymbolSpace.episodicConcepts` (default 4) and column `episodic_concepts` replace `episodicPercepts` and `episodic_percepts` with no alias; `EpisodicMemory.reinstate` publishes the kept concepts as a field directly (`_field` and the read through `cs_read_percept_field` go); `BasicModel.reverse` of that field is the recalled moment's reconstruction with no compose and no generate. Traced first: round 7's review already ends in an order-zero concept field. Claude's readings, flagged to Alec: the default stays 4; a review empties the querying stream's present percept readings. The Codex text is final with sixteen items.
- 2026-11-11 (Alec: "It is midway on work, but I'm happy to update midway through it's work."): Codex landed round 7 as `984a75b20e` and `a39cbfdef8` and is building the whole-path round; the text it landed already has the final XOR item, so the only update is item 16 (the episodic entry holds concepts). In the update text, "built first" in the line above is superseded by "at the next clean point". Checked: the scratch copy used for the XOR measurements equals `a39cbfdef8`'s `bin` file for file.

- 2026-11-11 (whole-path implementation): one greedy path plus `explorePaths` distinct departures, one R + A keep per stream, one reader and one sentence update; pre-commit thought with kept effects, recorded-output reporting (Alec's choice), adaptive input omission, and `episodicConcepts` with direct field review. This round is not committed or pushed.
- 2026-11-11 (Alec, on what belongs before the main run: "Let's discuss first, but also, all should go into the last architecture round (which should happen before optimization and final housekeeping)."): countdown item 1.8 added above with six contents and his words for each (a cheaper forgetting pass; System 1 as a grammar file of intersection, union and negation; an intuition may answer and reconstruct; trust 1 for percepts and 0 for new object-concepts; learning gated by operators in the grammar file; a query on concepts with no external `.when` or `.where`). Under discussion and not specified; the number 1.8 is Claude's. The whole-path round is not changed by it.
- 2026-11-11 (Alec, continuing the discussion of item 1.8): points 1, 3 and 4 "Good"; point 5 "fits well if we can decide on 2"; point 2 is now his proposal that a serial operand be one anchor in the conceptual hierarchy with all its conceptual parts and wholes, so that single operators act on a field (recorded under item 1.8, not decided: "a big change, let's discuss"). On the episodic entry he corrected item 16 of the whole-path round, which Codex had already built with pole pairs and brackets: "We should be storing order-0 concepts, not "brackets": what even are brackets?"; "a single signed value can be kept, as an approximation of the pair; dharmakirti says that contradiction is not a valid mental object, so presumably contradiction is a working-memory only." Its rules, certificates and item 16 amended; sent to Codex as the second mid-round update. Claude's readings, flagged to Alec: the signed value is for the entry's stored references only (rows' `truth_plus` and `truth_minus` stay a pair); a review lays its readings out from their order. Codex's progress note records one decision Alec gave it directly (recorded-output replay for the batch-end report).

- 2026-11-11 (the second mid-round update): episodic entries now keep reading-ordered concept references and one signed value each; stored pole pairs and places are retired, with named checkpoint migration. Recall lays out owned surfaces; missing surfaces and stale references are counted. Fields and semantic truth rows retain both poles.
- 2026-11-11 (Alec: "It sounds decided as item 1.8"): the last architecture round is decided in its seven contents (a cheaper forgetting pass; three field operators and the rest vector operators, told apart by the grammar file; an intuition may answer and reconstruct; trust 1 for percepts and 0 for new object-concepts; learning gated by enabling operators; a query on concepts with no store addressed by time or place; the loop as an examination phase that an animal file keeps). Not yet specified. Carried into its specification as Claude's defaults, each to be flagged there: the vector forms `conjunction` and `disjunction` give way to the field's and/or; the reconstruction walk stays costed as decided on 2026-11-10 and the projection onto the field is added beside it; operators are enabled by a mask over a grammar declared whole, so the graph does not change; the time-only recall gives way to the query on concepts. Checked the same hour on the live tree: Codex has applied the second mid-round update (`episodic_concepts` and `episodic_values`), and its return is still in progress.
- 2026-11-11 (Alec: "Let's write the 1.8 spec while waiting for codex. Let's not worry about who does 1.5 , 1, 0.5; we just want to work in parallel, with you doing most of the spec work."): the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) is written: eleven decisions in his words, the code as landed in round 7 read by three readers and spot-checked, ten rules, thirteen certificates, one measurement, and ten points open for him (its section 5). His corrections of the same message are in it: "Conjunction/disjunction stay symbolic, we generalize union/intersection to multiple-concept operations"; "Reconstruction is verbal from the conceptual field, or numeric from the conceptual field; these are now integrated"; "Numeric input can just be a linear layer onto concepts, the opposite of numeric output, so let's do it"; operators enabled by their stored trust under one file and the full grammar, their order inferred "from priority in a syntactic tree"; modifier plus noun as "an actual selection operator over all concepts in the extension of the count noun"; "Interpret trust is ... TBA"; "Linear layer might as well operate over all xor tests, as long as we have verbal tests that cover what they were doing symbolically". Two findings from the reading: the conjunctive definitions over concepts have a fold and no writer (`add_concept_edge` is never called with `conjunctive=True`), and `XOR_exact.xml`'s cases are promoted percept wholes. A toy run first: a clipped numeric input layer started at random solves XOR alone in 17 of 40 runs, never when started at the lane map $(x, 1 - x)$; conjunctions made by rule settle in 40 of 40. No Codex text yet.
- 2026-11-11 (Alec, on what trust `interpret` gives a definition: "Conceptual trust is just the certainty of the concept (its length), so interpret should somehow reflect the certainty of the mapping from the signifier to its signified. We might not be able to do that without direct evidence of the signified coming from a different modality; we might be able, therefore, to do it for the math training corpus; "one" is a symbol for the continuous value 1.0 . But multiple modality is definite future work."): open point 4 of the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) is answered and nothing changes in that round; [FutureWork](FutureWork.md#the-certainty-of-a-words-mapping-to-its-object-from-a-second-modality-alec-2026-11-11-future-work) has the new entry. Still on Claude's defaults there: who writes the intersection of several concepts, whether a composed point's field goes through the grammar's inverse, the order of operators from the syntactic tree, the verbal cover of the XOR proofs, and the standing convergence gate's move.
- 2026-11-11 (Alec, on the open points of the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md): "1 promotion on recurrence"; "The rest of the defaults are fine."): all ten points are answered and its section 5 now records them: intersections of several concepts are written by promotion on recurrence; a composed point's field goes through the grammar's inverse and reconstruction stays costed on it; operators are ordered from the syntactic tree by the rule of its section 3.7; the standing convergence gate moves to the numeric XOR file at the same bar; recall by time is removed. The specification has no open point. Its Codex text is written with the review of the whole-path round's return, on that round's code.
- 2026-11-11 (Alec: "Im actually leaning toward making numeric input/output a part of 1.8, so that interpret can have objects that it can perceive directly (e.g. interpret("one") = 1.0). it would be a toy example of multimodal for a 1-D (or N-D) numeric signal."): proposed in the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) as decision 13 (leaning, not decided), section 3.11 and certificate 14, with five points to settle in its section 5.1: a sentence may carry a row of numbers; the concept the numeric layer writes is a candidate object for a word of the same moment (`InterpretLayer.define` already takes evidence); the definition's trust is counted by views and agreement; under omission each modality restores the other; values in $[0, 1]$, one concept for each dimension, no arithmetic on the numbers, one lesson file.
- 2026-11-11 (Alec, on how the number enters: "percepts could probably operate on the value directly, and parts/wholes just becomes a less-than discrimination on the number line."; "so it would work for XR, but could generalize to images fairly easily."): the proposal in the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) section 3.11 now has a number as a basic percept whose presence is the value, part and whole between two as less-than (`Ops.partOf`), concepts read from it by the membership read with no learned input layer, and the XOR proofs getting their cases from percept wholes as `XOR_exact.xml` does. Two more points in its section 5.1: this is "amplitude is presence", which the decision of 2026-10-25 set aside for images ("Intensity is a code"); and whether the percept replaces section 3.9's learned input layer for the whole round. Not decided.
- 2026-11-11 (Alec: ".where would become 2D, and wholes and parts become hyperrectangles in the unit hypercube."; then the scope: "the image processing can be future word, and the ND stuff also, but the scalar case can be the continuous "feeling" that corresponds to the discrete boundaries established by symbolic discrimination."): the scalar case is in item 1.8 and images and signals of several dimensions are future work. The item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) decision 13 and section 3.11 now say so: one number may accompany a sentence; it is a basic percept whose presence is the value; a word that names it has the concept read from it as its object and adds the boundary; the XOR proofs read two readings of the scalar at two places. The image notes are at the head of FutureWork's image-modality entry. Six points of its section 5.1 still stand on Claude's defaults.
- 2026-11-11 (Alec: "so feelings are continuous, concepts are discrete, and we have a better model of nonconceptual knowing: continuity. We might also want some actual mathematical operations in this space: less than, multiplication, addition, min, max that can operate on the values directly rather than operating on the codes as proxies offerred by the codebook."): recorded in the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) (decision 13, section 3.11), in [Philosophy](Philosophy.md) under feelings, and as a new [FutureWork](FutureWork.md#operations-on-continuous-values-sum-and-product-on-the-feeling-alec-2026-11-11-to-settle) entry. Less-than, least and greatest come with the scalar (`Ops.partOf`, `intersection`, `union`) and are in certificate 14. A product and a sum on values are not in the round by default; open with Alec: which sum, and how they stand with the rule of 2026-10-07 that the math cannot be direct.
- 2026-11-11 (Alec: "Maybe part, max, and min, so that we don't have to invent overflow rules. That will at least give us a fuzzy ordering and norm/t-norm."): the operations on the scalar's value in item 1.8 are part, max and min, which come with the scalar (`Ops.partOf`, the field's `union` and `intersection`); no sum and no product on values is built, and the rule of 2026-10-07 on arithmetic is untouched ([specification](plans/2026-11-11-item1-8-last-architecture-round.md) decision 13, section 3.11, section 5.1 item 8). Still on Claude's defaults there: section 5.1 items 2 to 7.
- 2026-11-11 (Alec: "Your defaults are fine for this. But lets imagine tying this set of percepts to the existing text. If we provide both perceptual spaces to a single conceptual space for its integration, then we can ask questions like which is more or less, and have interpret() result in bindings to the other modality. That means that we can do things like teach number recognition just like large AIs do object recognition in photos."): the scalar's six remaining defaults are accepted and recorded as answered (item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) section 5.1). His picture is in decision 13 and section 3.11: two perceptual spaces give one conceptual space its evidence (it already reads two sources, the part tower and the whole tower); recognition is the definition read from the object to its word and said by the output path; comparison is the part test on two readings, its question's words learned from the answers. One point open (section 5.1 item 9, Claude's proposal): recognition on the trained values is in the certificate; recognition of untrained values and comparison are measured and reported.
- 2026-11-11 (Alec: "although we can represent 1,000,000 numerically, and we might not know what 1M amount of a feeling would feel like."): recorded in the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) as where the scalar's grounding stops (decision 13; section 3.11): it grounds only the words whose values can be felt, within a bounded range; every other numeral is known through other words, by the successor and by thinking, so the rule of 2026-10-07 stays.
- 2026-11-11 (Alec: "so the mapping of feeling to words is not linear."): recorded in the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) (decision 13; section 3.11). Nothing there assumes a linear mapping: a word's concept reads the scalar through its own learned exponent in `SparseLayer.fold_presence`, and the measurement on untrained values shows the shape.
- 2026-11-11 (Alec: "does this in fact give us enough for a primitive example of object recognition?"; "or symbolization? can the model learn the word for a number (where that word is a concept with part and whole bounds that are hyperrectangles)?"): answered in the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) section 3.11 as a proposal (its section 5.1 item 10): a word's concept over the scalar is a box whose lower and upper bounds are the least and the greatest of the values heard with the word (`intersection` and `union` over the definition's views), with membership by the two coverages of `meet_ratios`; only part, min and max are used and nothing is trained by gradient. A trace made for the answer revises an accepted default (section 5.1 item 2): `InterpretLayer.lexical_form` spells an object only where exactly one word names it, so "one" and "zero" are two concepts and not two lanes of one. Not traced: what makes the perceived concept the thing that is said.
- 2026-11-11 (Alec: "No, percepts do not need two lanes, that is just for concepts."): the second pole that Claude had put on the scalar's percept (carrying $1 - v$; accepted with the other defaults) is withdrawn, and so is Claude's reading of a concept's two lanes as a box's bounds. It is Philosophy's "Perception carries no negation". What follows is written into the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) section 3.11 as Claude's proposal (section 5.1 item 11): with one lane a value of 0 is no feeling at all, so a word for it names an absence, inferred on a concept's against-lane and never perceived; the toy's two words become "half" at .5 and "one" at 1.0, both felt; the XOR proofs' two symbols become the levels .5 and 1.0, as `XOR_exact.xml`'s are two letters; "zero" is a third lesson, measured.
- 2026-11-11 (Alec: "XOR is traditionally done in 2d, so 0 and 1 would be fine for the input values."): the XOR proofs of item 1.8 read an input of two dimensions with the values 0 and 1 ([specification](plans/2026-11-11-item1-8-last-architecture-round.md) section 3.11; section 5.1 item 11). Claude's proposal of two felt levels for them is withdrawn, with its worry: no absence has to be inferred, since the numeric output is linear with signed weights and XOR is $x_1 + x_2 - 2\,(x_1 \wedge x_2)$; the one concept needed is the intersection of the two inputs, written by promotion when they recur together, and without it the file stays at $\tfrac14$. Checked by a small computation: weights 1, 1 and -2 exactly; the control's error .25; a linear output from random starts settles in 40 of 40 runs. The old read-out's "two positive edges" assertion is to be ported with a record.
- 2026-11-11 (Alec: "Please write the 1.8 spec , I will let you know when codex finishes the next 2.0 round"): the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) is consolidated into one text. Nineteen decisions in his words; the code as landed in round 7; eleven rules (3.9 is numbers in and out, with a number as a one-lane percept, part, max and min on values, a linear output on concepts, and the XOR proofs on two inputs of 0 and 1; 3.10 is the scalar beside the words; 3.11 the forgetting pass); fourteen certificates and one measurement; and its section 5 with what was answered, three defaults he has not spoken to (the box as how a word's concept over the scalar is learned; recognition required while untrained values, comparison and "zero" are measured; the toy's words "half" and "one") and one link not traced (what makes the perceived concept the thing said). Section numbers in the lines above of this date refer to the first writing, a copy of which is in Claude's scratch directory and not in the repository. No Codex text yet: it is written with the review of the whole-path round's return.
- 2026-11-11 (Alec: "We might as well allow numeric values in 0 to 1 range, but map that to "zero" to "infinity", so that we can represent all numbers, and let the scale sort itself out with some sort of spacing algorithm (so its an ordinal dim that becomes approximately real with increased ordinates)."): decision 20 of the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md). A number $n$ is felt as $f(n)$ in $[0, 1]$, increasing, zero at zero and one at infinity; part, max and min are unchanged by any such $f$; a word's box is kept on numbers and read through $f$; the toy's words become "one", "two" and "three". Open (its section 5.2 e): $f$ fixed (for instance $n/(1+n)$) or the running rank of $n$ among the numbers met, Claude's recommendation.
- 2026-11-11 (Alec: "Codex is ready for review."): the whole-path round's return is reviewed and accepted. The live tree hashes to the returned `26dd5a0bda60`; Claude re-ran the stopping gates on it (seven production files changed after the frozen sweep): convergence, the shared-concept case, the grammar reconstruction gate and the round's 26 new certificates, 29 passed. The XOR gate passed five of ten, exactly the bar. Two readers checked the code against the rules; five defects were confirmed in the code and are repaired first in the next text, none touching this return's measurements: omission leaks into the percept store, the document address and `reconstruction.free_percept`; the batch end keeps the last sentence slot for every row of a packed batch; a thought departure restores every stream's state; with two or more explore paths each compose credit reads the last path's state and the lessons are unscaled. One reader's claim (the output residual weighted twice) was checked and is not so. Warts: the lessons say nothing (4, 0 and 1 nonempty answers of 80), omission never engaged, the thought store grows at run time (for item 1). The landing opens item 1.8's Codex text, which waits for 1.8 section 5.2 d (recognition: a binding to a perceived concept as an answer) and e (the spacing of the scale).
- 2026-11-12 (Alec, on item 1.8's two open points; spacing: "Running rank (Recommended)"; recognition: "You seem to be asking about identity. I had theorized that order-0 events could not be nouns, because generalizing over events could yield only a general event. But that serves in MTM as an order-0 identity, and is used by order 0 as a promotion criteria."): the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) has decision 21 and the spacing rule (the running rank of a number among the numbers met, a bounded sorted record updated over a horizon). Claude's reading of the recognition answer, for his word (its section 5.2 d): "this" points at the present event at order zero, held in MTM as its order-zero identity; the question asks for the class whose member the event is, an identity of order one or more such as the word's concept over the scalar; `bind_site` adds the individuals whose evidence in the moment's field contains the reading, and such a binding is accepted as perception. The 1.8 Codex text waits only for that confirmation.
- 2026-11-12 (Alec: "My understanding of "running rank" is a linear interpolation over recognized concepts, from 0-1."): item 1.8's spacing rule rewritten ([specification](plans/2026-11-11-item1-8-last-architecture-round.md) section 3.10, section 5.2 e): the recognized concepts over the scalar, by their numbers, are spaced evenly over the unit interval with zero at 0 and infinity at 1, and $f$ is linear between neighbours; Claude's description (the rank among all numbers met, over a horizon) is replaced. Claude's, for the end his rule leaves open: above the largest recognized number $f$ is linear in $1/n$, so that it reaches 1 at infinity. With recognized numbers 1 to $N$ the scale is linear over that range; with the numbers a language names often it is close to logarithmic.
- 2026-11-12 (Alec: "Also, for the numeric modality, please confirm if we can use the values as the numerical codebook codes directly, rather than creating arbitrary codes, it would seem to be more efficient."): confirmed, and written into the item 1.8 [specification](plans/2026-11-11-item1-8-last-architecture-round.md) as decision 22 and section 3.9: a numeric dimension has one atom whose basis row is a fixed axis of its own band, and a reading's code is $f(n)$ on that axis; no row per value, no drawn pattern, no collision rule. It is what makes part, max and min the code operations themselves (`Ops.partOf` is elementwise less-or-equal; join and meet are max and min). One consequence recorded: every value of a dimension has the same direction, so readers that identify by cosine see one identity, the dimension; values are told apart only by order and containment.
- 2026-11-12 (Alec: "If that is the case, then we should do something like a logarithmic mapping from 0..INF to 0..1"): item 1.8's spacing is a fixed logarithmic map ([specification](plans/2026-11-11-item1-8-last-architecture-round.md) decision 20, section 3.10, section 5.2 e), replacing the interpolation over recognized concepts: $f(n) = g/(1+g)$ with $g = \ln(1+n)$, Claude's form for "something like" (the logarithm alone has no upper end). One is .409, ten .706, a million .933; single precision still tells $n$ from $2n$ at $10^{100}$ (where $n/(1+n)$ stops near $10^8$). It has no state and nothing moves when a concept is recognized. The XOR proofs read 1 as .409; the linear output takes the scale.
- 2026-11-12 (Alec: "Confirming also that order-zero events are suitable for order-zero identity. I thing we already decided that MTM is ramsified, so this would be a necessary consequence"): item 1.8's recognition rule is settled ([specification](plans/2026-11-11-item1-8-last-architecture-round.md) decision 21, section 5.2 d): "this" binds to the present event's order-zero identity in MTM, the question's open reference binds to the class whose member the event is, and the binding is accepted as perception. It amends the constraint of 2026-11-03 that no order-zero identity is opened; Claude's reconciliation: succession makes a sequence, the same reading met again makes an order-zero identity. The specification's two references to `LanguageLayer.set_stage_menu` are corrected to `OperationSelectionLayer`.
- 2026-11-12: item 1.8's Codex text is written, final ([specification](plans/2026-11-11-item1-8-last-architecture-round.md) section 7, return as its section 8): steps 1 to 3 land the whole-path round (returned source `26dd5a0bda60`), item 4 repairs the five defects of the whole-path review, items 5 to 16 build item 1.8, item 17 is the certificates and the one measurement. Checked before handing it over: every code name it cites exists in the returned source; certificate 13 and the scalar's measured probes were brought in line with the scale of decision 20 (words for one, two and three; untrained numbers .5 to 3.5 and 10).
- 2026-11-12 (Alec, when the item 1.8 text was handed over: "Now? We have not completed and pushed 2.0"): item 2.0 is completed by its own Codex text: land the accepted whole-path round and push, then repair the five defects found in its review, with the stopping gates, the XOR class certificate once (reported) and one make test, and no measurement campaign. Item 1.8's text (its specification's section 7) is held: it now opens by landing those repairs, whose hash is filled in at their review, and its item 4 moved to the completion text.
- 2026-11-12 (Alec: "You need to review."): nothing new from Codex (no landing, no return; the source is still `26dd5a0bda60`), so Claude reviewed its own item 2.0 completion text, which it had handed over unchecked. Three corrections, recorded under the text: the list of documents changed since the return; the omission certificate's edit named by content; and the XOR class certificate measured by its criterion rather than run under `RUN_TRAINED=1`, where it fails for its absent phase artifact. The text is final again.

Landed as b35a4bd3167049b9ef3559104d478d89fd869e80, 2026-11-12

- 2026-11-12 (item 2.0 completion): all five whole-path review findings and the shared-registry finding are repaired at their producers, with 42 targeted cases passing on `a22afbd67871f`. Alec authorized committing and pushing these repairs as 2.0 before any 1.8 work, superseding the earlier hold for their review. Adequate targeted testing now; the full standard suite after 1.8, with `RUN_SLOW` unset. No item 1.8 implementation is included.

- 2026-11-12: item 1.8 candidate returned through Alec for Claude's review in [§8](plans/2026-11-11-item1-8-last-architecture-round.md#8-implementation-return-2026-11-12-through-alec-for-claudes-review), source `607ab7f8c5da`. Both stopping gates, 83 focused cases and ten native numeric XOR trainings passed. The single standard sweep completed 8,096 cases: 7,595 passed, 177 failed, 323 skipped, 1 expected failure. Symbolic reconstruction and full scalar recognition/omission fail; the lesson stopped at the query capability grant, and measurement gaps are recorded. **Not ready to land.** The item 2.0 repair landing remains `9f7923d67f89`; no item 1.8 commit or push, and no source edits after the freeze.

Everything older than this list, and the full text of what was removed from this file on 2026-11-26, is in [DevelopmentHistory](DevelopmentHistory.md#work-recorded-in-todomd-until-2026-11-26).
