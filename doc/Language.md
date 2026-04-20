# Language

## Overview

The language system runs a single unified **S-tier** grammar over symbol
activations.  One class, `SyntacticLayer`, owns both rule prediction
(Gumbel-softmax over rule logits) and rule semantics (the `<name>Forward`
/ `<name>Reverse` bodies in the `_RULE_METHODS` dispatch table).  After
the 2026-04-19 merge there is no longer a separate
`PerceptualSyntacticLayer` / `ConceptualSyntacticLayer` /
`SymbolicSyntacticLayer` trio -- `WordSpace` instantiates a single
`syntacticLayer` and SymbolicSpace calls it.

```
InputSpace  ->  PerceptualSpace  ->  ConceptualSpace  ->  SymbolicSpace  ->  OutputSpace
                                                     grammar runs here
                                                     17 S-tier rules
```

Prior revisions partitioned rules across three tiers (S, C, P) with
transitions $S \to C \to P$.  The 2026-04 redesign collapsed C into S
and removed the P-tier syntactic layer (chunking became a non-syntactic
pre-pass).  All grammar rules now live on S-tier and execute on symbol
activations; the perceptual chunking step is handled by `ChunkingLayer`
outside of syntax.

---

## Grammar

`TheGrammar` is a singleton `Grammar` instance.  Under `MentalModel.xml`
it is configured with 17 S-tier rules (IDs 0-16):

| Rule | Production | Arity | Method |
|------|-----------|-------|--------|
| 0 | $S \to \text{true}(S)$ | 1 | `trueForward` |
| 1 | $S \to \text{false}(S)$ | 1 | `falseForward` |
| 2 | $S \to \text{non}(S)$ | 1 | `nonForward` |
| 3 | $S \to \text{conjunction}(S,\ S)$ | 2 | `conjunctionForward` |
| 4 | $S \to \text{disjunction}(S,\ S)$ | 2 | `disjunctionForward` |
| 5 | $S \to \text{what}(S)$ | 1 | `whatForward` |
| 6 | $S \to \text{where}(S)$ | 1 | `whereForward` |
| 7 | $S \to \text{when}(S)$ | 1 | `whenForward` |
| 8 | $S \to \text{query}(S,\ S)$ | 2 | `queryForward` |
| 9 | $S \to \text{swap}(S,\ S)$ | 2 | `swapForward` |
| 10 | $S \to \text{equals}(S,\ S)$ | 2 | `equalsForward` |
| 11 | $S \to \text{not}(S)$ | 1 | `notForward` |
| 12 | $S \to \text{part}(S,\ S)$ | 2 | `partForward` |
| 13 | $S \to \text{intersection}(S,\ S)$ | 2 | `intersectionForward` |
| 14 | $S \to \text{union}(S,\ S)$ | 2 | `unionForward` |
| 15 | $S \to \text{lower}(S,\ S)$ | 2 | `lowerForward` |
| 16 | $S \to \text{lift}(S,\ S)$ | 2 | `liftForward` |

The dispatch table also knows `chunk` (an invertible max-union used by
the perceptual pre-pass), but `chunk` is not registered in the default
S-tier rule list.

### Accessors

- `Grammar.symbolic()` - indices of all S-tier rules
- `Grammar.symbolic_transition()` - `None` (no cross-tier transitions)
- `Grammar.binary_rules()` - indices of all arity-2 rules
- `Grammar.rule_by_id(i)` - canonical production string
- `Grammar.method_name(i)` - dispatch method name (e.g. `'swap'`)
- `Grammar.arity(i)` - 1 or 2
- `Grammar.tier(i)` - `'S'`

### Grammar Configuration

`Grammar.configure()` loads an XML `<grammar>` section (inside `<WordSpace>
<language>`) and parses **only** `<S>` entries:

```xml
<grammar>
  <S>true(S)</S>
  <S>swap(S, S)</S>
  <S>equals(S, S)</S>
  ...
</grammar>
```

Rules are specified in functional notation: `swap(S, S)`, `not(S)`,
`intersection(S, S)`, etc.  Non-`<S>` keys (legacy `<C>`, `<P>`) are
silently dropped during parse.  The XSD at `data/model.xsd` enforces
the S-only structure.

### Thought-Free Mode (Shamatha Speech)

`Grammar.thought_free` is a boolean flag (default `False`) settable
per-request by WikiOracle's `serve.py` before inference and reset in a
`finally` block.  Under the old three-tier grammar it suppressed C-level
`not`; after the S-merge its filtering effect is minimal pending
redesign.  See [Params.md](Params.md) for the XML configuration.

---

## Rule Algorithms (LaTeX)

Let $x, \ell, r \in \mathbb{R}^{B \times N}$ (activation mode) or
$\mathbb{R}^{B \times N \times D}$ (vector mode).  Let $\mathcal{B}$ be
the subspace basis (e.g. a bitonic `Basis`), which supplies pointwise
primitives `pos`, `negation`, `conjunction`, `disjunction`, `equal`,
`part`.  When $\mathcal{B}$ is absent the methods fall back to torch
elementwise primitives.  `mask` (optional concept-axis `Mask`) gates the
output by zeroing non-selected dimensions; it is omitted below for
clarity.

### Trinity partition (true / false / non)

For $x \in [-1, 1]$ (clamped defensively on entry):

$$
\begin{aligned}
\mathrm{true}(x)  &= \mathcal{B}.\mathrm{pos}(x)   = \max(0,\ x) \\
\mathrm{false}(x) &= \mathcal{B}.\mathrm{pos}(-x)  = \max(0,\ -x) \\
\mathrm{non}(x)   &= 1 - |x|
\end{aligned}
$$

These three operators partition unity: $\mathrm{true} + \mathrm{false}
+ \mathrm{non} = 1$ pointwise.  The ReLU shape of `true`/`false` makes
them the "committed yes" and "committed no" halves of a bitonic axis;
`non` is the triangular indeterminate residual peaked at $x = 0$.  All
three are lossy (no inverse registered).

### Negation (not)

$$
\mathrm{not}(x) = \mathcal{B}.\mathrm{negation}(x) = -x,
\qquad
\mathrm{not}^{-1} = -x
$$

Pure antipodal flip on the bitonic axis.  Self-inverse.

### Conjunction / disjunction

$$
\mathrm{conjunction}(\ell, r) = \mathcal{B}.\mathrm{conjunction}(\ell, r)
                              \ \xrightarrow[\text{fallback}]{}\ \min(\ell, r)
$$

$$
\mathrm{disjunction}(\ell, r) = \mathcal{B}.\mathrm{disjunction}(\ell, r)
                              \ \xrightarrow[\text{fallback}]{}\ \max(\ell, r)
$$

Sentence-level fuzzy $\wedge$ and $\vee$ on bitonic activations.  Both
are lossy at the sentence level (no inverse registered for the S-tier
usage), though the same primitives at C-tier subspace level retain
Basis-level inverses.

### Intersection / union (mereological)

$$
\mathrm{intersection}(\ell, r) = \mathcal{B}.\mathrm{conjunction}(\ell, r)
\qquad
\mathrm{union}(\ell, r) = \mathcal{B}.\mathrm{disjunction}(\ell, r)
$$

Inverses are the Basis inverses on $r$:

$$
\mathrm{intersection}^{-1}(y, r) = \mathcal{B}.\mathrm{conjunction\_inverse}(y, r)
\qquad
\mathrm{union}^{-1}(y, r) = \mathcal{B}.\mathrm{disjunction\_inverse}(y, r)
$$

### Part (clipped cosine)

Parthood score, weighting the right operand by its containment in the
left:

$$
\mathrm{part}(\ell, r) = \mathcal{B}.\mathrm{part}(\ell, r,\ \mathrm{scalar}{=}\mathrm{True})
\cdot r
= \frac{\max(0,\ \ell \cdot r)}{|\ell|\,|r|} \cdot r
$$

The scalar $s \in [0, 1]$ is broadcast to $r$'s rank before
multiplication.  Empty-operand contract: if $|\ell|$ or $|r|$ is near
zero, $s = 1$ (the empty set is part of everything).  Lossy.

### Equals (cosine-gated projection)

$$
\mathrm{equals}(\ell, r) = \mathcal{B}.\mathrm{equal}(\ell, r,\ \mathrm{scalar}{=}\mathrm{True})
\cdot r
$$

"Pass $r$ upward only to the degree $\ell$ matches it."  The equal
score is mutual parthood
$\mathcal{B}.\mathrm{equal}(\ell, r) = \mathrm{part}(\ell, r)\cdot\mathrm{part}(r, \ell)$,
scalar-reduced and broadcast to $r$.

Under a `mask`, equals degenerates to an L1-distance agreement score on
the selected dims:

$$
\mathrm{equals}^{\mathrm{mask}}(\ell, r)
= \mathrm{clip}\!\left(1 - \frac{\sum_d m_d\,|\ell_d - r_d|}{\sum_d m_d},\ 0,\ 1\right) \cdot r
$$

When the owning `SymbolicSpace` supplies a `PiLayer` back-reference and
both operands match the Pi output dim, `equalsForward` first reverse-
projects operands to concept space and invokes the concept-basis `equal`
on the bitonic subspace.  Otherwise it uses the local basis.  Lossy.

### Slot selectors (what / where / when)

Given per-subspace column widths $(n_{\text{what}}, n_{\text{where}}, n_{\text{when}})$:

$$
\mathrm{what}(x)[\ldots, i] =
\begin{cases}
x[\ldots, i] & 0 \le i < n_{\text{what}} \\
0 & \text{otherwise}
\end{cases}
$$

$$
\mathrm{where}(x)[\ldots, i] =
\begin{cases}
x[\ldots, i] & n_{\text{what}} \le i < n_{\text{what}} + n_{\text{where}} \\
0 & \text{otherwise}
\end{cases}
$$

$$
\mathrm{when}(x)[\ldots, i] =
\begin{cases}
x[\ldots, i] & n_{\text{what}} + n_{\text{where}} \le i \\
0 & \text{otherwise}
\end{cases}
$$

Parameter-free axis projections.  In 2D (`[B, N]`) activation mode the
block structure is unavailable, so selectors degenerate to identity --
the axis semantics are carried by the subspace's modality tensors
rather than the flat activation vector.  Lossy.

### Query (accumulator preservation)

$$
\mathrm{query}(\ell, r) = \ell
$$

The query marker is pushed onto WordSubSpace at the accumulation point
by `compose()` when it detects norm-drop (see below), not by
`queryForward` itself.  When the parse tree is re-evaluated downstream,
$\mathrm{query}$ returns the preserved accumulator (the left operand).
Lossy.

### Swap (Sinkhorn soft permutation)

Let $M \in \mathbb{R}^{3 \times 3}$ be a learned logit matrix.  Apply
$k = 5$ Sinkhorn normalisation iterations to produce a doubly-stochastic
soft permutation $P$:

$$
M^{(0)} = M, \qquad
M^{(t+1)} = M^{(t)} - \mathrm{logsumexp}_{\mathrm{cols}}(M^{(t)}) - \mathrm{logsumexp}_{\mathrm{rows}}(M^{(t)}),
\qquad
P = \exp(M^{(k)})
$$

Given operands $\ell, r$ and a learned broadcast marker $m$, stack and
contract along the three "slots":

$$
\mathrm{swap}(\ell, r) = \bigl(P \cdot [\ell,\, r,\, m]^\top\bigr)_0
$$

i.e.\ the first row of the mixed-slot stack.  The permutation is shared
across batch / position / dim.  Lossy.

### Lift / lower (in-space algebra)

Post 2026-04-19 the old PiLayer round-trip (forward to S, combine,
reverse to C) collapses to in-space binary algebra because the caller
already holds the forwarded operands:

$$
\mathrm{lift}(\ell, r) = \ell \odot r,
\qquad
\mathrm{lift}^{-1}(y, r) = y \oslash (r + \varepsilon)
$$

$$
\mathrm{lower}(\ell, r) = \tfrac{1}{2}(\ell + r),
\qquad
\mathrm{lower}^{-1}(y, r) = 2\,y - r
$$

Both registered with inverses.  The legacy `LiftingLayer` /
`LoweringLayer` classes still exist in `Layers.py` for `TruthLayer`'s
universality scoring; they are no longer instantiated per SyntacticLayer.

### Chunk (invertible max-union; pre-pass)

$$
\mathrm{chunk}(\ell, r) = \mathcal{B}.\mathrm{disjunction}(\ell, r,\ \mathrm{monotonic}{=}\mathrm{True})
\ \xrightarrow[\text{fallback}]{}\ \max(\ell, r)
$$

Monotonic max-union.  Inverse via
$\mathcal{B}.\mathrm{disjunction\_inverse}(y, r)$.  Not registered in the
default S-tier rule list; used by `ChunkingLayer` in the perceptual
pre-pass.

---

## Rule Prediction (`forward`)

`SyntacticLayer.forward(x)` predicts a rule distribution at each
derivation depth.  The architecture is weight-tied (recursive) across
depths with a learned depth embedding.

Input projection:

$$
h^{(0)} = \sigma(W_{\text{in}}\, x + b_{\text{in}}),
\qquad h^{(0)} \in \mathbb{R}^{B \times d_{\text{hidden}}}
$$

For each depth $d = 0, 1, \ldots, D_{\max} - 1$ (shared weights
$W_{\text{deriv}}$, $W_{\text{head}}$):

$$
\begin{aligned}
h^{(d+1)}  &= \sigma\!\bigl(W_{\text{deriv}}\,(h^{(d)} + e_d) + b_{\text{deriv}}\bigr) \\
z^{(d)}    &= W_{\text{head}}\,h^{(d+1)} + b_{\text{head}}                \in \mathbb{R}^{B \times K} \\
z^{(d)}_{t}&\leftarrow \mathrm{stop\_grad}\bigl(z^{(d)}_{t}\bigr) + (1 - \iota)\cdot \tau_{\text{trans}}
           \quad \text{(transition-bias on rule } t\text{)} \\
p^{(d)}    &=
\begin{cases}
\mathrm{gumbel\_softmax}(z^{(d)}; \tau) & \text{training} \\
\mathrm{softmax}(z^{(d)})               & \text{eval}
\end{cases}
\end{aligned}
$$

where $e_d$ is the depth embedding, $\iota = \mathrm{grammar.interpretation}$,
$t$ is the index of the transition rule (if present), and $\tau$ is the
Gumbel-softmax temperature (annealed from 1.0 toward 0.1 via `set_tau`).

Output: `{rule_logits, rule_probs, predicted_rules, words}`.

### Differentiable rule selection

- **Training:** `gumbel_softmax(logits, tau)` -- soft one-hot,
  gradients flow through all candidate rules proportional to probability.
- **Eval:** `softmax` + `argmax` -- discrete rule selection.
- **Annealing:** $\tau$ starts at 1.0 (diffuse) and decreases toward
  0.1 (near-discrete) over training.

---

## Projection: `project(grammar, rule_id, left, right)`

Rule dispatch flows through `SyntacticLayer.project()`.  It looks up
`(forward_name, reverse_name, is_binary)` in
`_RULE_METHODS[method_name]` and calls the named `<name>Forward`.
Inputs are `[B, N]` activations (2D mode) or `[B, N, D]` bivectors
(3D mode).

`reverse_project` is the dual, used during `decompose` to invert the
registered reversible rules.

---

## Composition: `compose(data, subspace, grammar, target_count=None)`

`SyntacticLayer.compose()` is the unified two-phase driver.  It
dispatches on input rank:

- `data.ndim == 2`  -->  `_compose_activation` (2D activation mode)
- `data.ndim == 3`  -->  `_compose_vector`     (3D bivector mode,
  with optional pairwise reduction via `_compose_to_target`)

Returns `(composed, svo_or_None)`; `svo` is currently always `None`.

### Phase 1: deterministic `not`

Before soft superposition (3D path only), apply a single deterministic
`not` to the top-of-stack when that vector has a negative mean:

$$
\text{for each batch } b \text{ with top-of-stack position } p_b:
\qquad
\text{if } \overline{\mathrm{data}[b, p_b]} < 0
\implies \mathrm{data}[b, p_b] \leftarrow \mathrm{not}(\mathrm{data}[b, p_b]),
$$

and a `not` word is pushed for $b$ at position $p_b$.  This strips any
leading negation from the accumulator before it mixes with siblings.

### Phase 2: soft-weighted cascading composition

Let $v^{(b)}_i \in \mathbb{R}^D$ be the $i$-th active leaf vector in
batch $b$, and let
$L_b = |\{\text{active positions in } b\}|,\ L_{\max} = \max_b L_b$.

Build leaf slabs $v[\cdot, i, \cdot, \cdot]$ of shape
$[B, L_{\max}, N, D]$ by masking.  Initialize the accumulator:

$$
a^{(0)} = v[\,\cdot\,,\ 0\,,\,\cdot\,,\,\cdot\,]
$$

Let $\mathcal{R}^\star$ be the *composable* rule set: the grammar
S-rules minus `not` (already applied in Phase 1).  Let
$p^{(d)} \in \Delta^{|\mathcal{R}^\star|}$ be the rule-prediction
probabilities for depth $d$ renormalised over $\mathcal{R}^\star$.

Iterate depth $d = 0, 1, \ldots$ until $d \ge D_{\max}$ or the next
leaf index exceeds $L_{\max}$:

$$
\ell = a^{(d)},
\qquad
r = v[\cdot,\,d + 1,\,\cdot,\,\cdot]
$$

$$
\text{for each } \rho \in \mathcal{R}^\star:
\qquad
u^{(d)}_\rho =
\begin{cases}
\rho(\ell, r) & \mathrm{arity}(\rho) = 2 \\
\rho(\ell)    & \mathrm{arity}(\rho) = 1
\end{cases}
$$

Soft candidate (training) vs. argmax candidate (eval):

$$
c^{(d)} =
\begin{cases}
\sum_{\rho} p^{(d)}_\rho\, u^{(d)}_\rho & \text{training} \\
u^{(d)}_{\rho^\star},\quad \rho^\star = \arg\max p^{(d)} & \text{eval}
\end{cases}
$$

#### Query / norm-drop detection

Detect symbolic contradiction at the accumulation point by comparing
the pre- and post-composition Frobenius norms per batch row:

$$
\nu_\ell = \left\|\ell\right\|_F,
\qquad
\nu_c = \left\|c^{(d)}\right\|_F,
\qquad
\mathrm{qmask}_b = \bigl(\nu_{\ell,b} > 10^{-6}\bigr) \wedge
                   \bigl(\nu_{c,b} < \kappa\,\nu_{\ell,b}\bigr)
$$

with $\kappa = 0.1$ (`_QUERY_NORM_DROP_RATIO`).  If $\mathrm{qmask}_b$
fires, the new accumulator preserves the prior state and a $\mathrm{query}$
word is pushed onto WordSubSpace with leaves
$(\mathrm{last\_rule}_b,\ \arg\max_\rho p^{(d)}_b,\ -1)$; otherwise
accept the candidate:

$$
a^{(d+1)}_b =
\begin{cases}
\ell_b    & \mathrm{qmask}_b = 1 \\
c^{(d)}_b & \mathrm{qmask}_b = 0
\end{cases}
$$

The argmax rule is recorded as a word at the leaf position, and
`last_rule_per_batch[b]` is updated iff the query mask did **not** fire
(so `query` preserves the prior rule context).

Advance $d \leftarrow d + 1$ and
$\text{leaf\_idx} \leftarrow \text{leaf\_idx} + 1$ (or $+2$ for arity-3
rules when a third leaf is available).

### Leaf ledger (codebook indices)

Before Phase 2 begins, codebook indices of the original leaves are
snapshotted:

$$
\text{cb\_idx}[b, i] = \arg\max_k\,\bigl(\mathrm{data}[b, i] \cdot \mathrm{cb}[k]^\top\bigr)
$$

and each active leaf emits a transition word with `order = -1` and
`leaf1 = cb_idx[b, pos]`.  This ledger is what `decompose` uses to
reconstruct the pre-compose tensor exactly.

### 2D activation path (`_compose_activation`)

The 2D path runs the same depth-wise mixture but on `[B, N]` scalar
activations:

$$
a^{(d+1)} = \sum_\rho p^{(d)}_\rho \cdot
\begin{cases}
\rho(a^{(d)}, v[\,\cdot\,,\,d+1])    & \mathrm{arity}(\rho) = 2 \\
\rho(a^{(d)})                         & \mathrm{arity}(\rho) = 1
\end{cases}
$$

with the same query/norm-drop logic (norm over the $N$-axis instead of
the $[N,D]$-plane).  This path is used when the caller forwards norms
directly (legacy S-tier driver).

---

## Pairwise reduction: `_compose_to_target(data, subspace, grammar, target_count)`

When `target_count` is supplied, the 3D path reduces the active-token
count down to that value via iterated pairwise reductions.  Composable
rules in this path are the arity-2 rules minus `not`.

At each outer iteration $d$:

1. Count active positions per batch.  Stop if $\max_b |A_b| \le
   \text{target\_count}$.
2. For each batch, walk pairs $(p_{2i}, p_{2i+1})$ while reduction
   would still exceed the target.  For each pair, compute all
   composable rule outputs, take either the soft mixture (training) or
   argmax (eval), write the result to the left position, zero the
   right, and append the left position to the new active list.
3. Record the chosen rule as a word at `left_pos` with the current
   outer-iteration order.
4. Carry forward any leftover unpaired positions.

Finally, extract the non-zero subset of `data` at the surviving
positions, zero-fill the rest, and return.

---

## Decomposition: `decompose(data, subspace, grammar)`

Reconstruct the pre-compose tensor from the symbolic word record.

- If a codebook is available and shapes match, look up each terminal
  word (those with `order = -1`) and place `cb[leaf1]` at the recorded
  position, producing the exact pre-compose tensor.
- Otherwise, fall through to the degraded rule-reversal pass: walk
  words in reverse and invert the registered reversible rules
  (`not`, `intersection`, `union`, `lift`, `lower`).  Lossy rules
  (`equals`, `part`, `true`, `false`, `non`, `what`/`where`/`when`,
  `query`, `swap`, `conjunction`, `disjunction`) pass through.

---

## Two-Layer Logic

The logic system operates at two levels:

### Subsymbolic Layer (Vectors)

Objects are vector sets $(B, N, D)$ interpreted as RBF / luminosity
fields.

| Operator | Implementation | Meaning |
|----------|---------------|---------|
| union | $\max(l, r)$ | strongest affirmation |
| intersection | $\min(l, r)$ | shared commitment |
| negation (not) | $-x$ | antipodal opposition on hypersphere |
| non | $\alpha x,\ \alpha \in [0,1)$ | contraction toward zero (withdrawal of assertion) |
| parthood | fuzzy max-coverage $\in [-1,1]$ | signed containment |

### Symbolization

Map vectors -> scalar truth strength:

$$
s(X) = 2 \cdot \mathrm{mean}(\|x_i\|) - 1 \quad \in [-1, 1]
$$

Interpretation: +1 -> strong presence, 0 -> neutral, -1 -> absence.

### Symbolic Layer (Scalars in [-1,1])

Let $a, b \in [-1,1]$.

| Operator | Formula | Meaning |
|----------|---------|---------|
| neg | $-a$ | oppositional negation |
| non | $\alpha a$ | withdrawal / neutrality |
| union | $\max(a, b)$ | strongest affirmation |
| intersection | $\min(a, b)$ | shared commitment |
| part | $\operatorname{clamp}(b - a, -1, 1)$ | signed containment |

The key insight: subsymbolic = geometry, symbolic = order + polarity,
symbolization = norm projection.

---

## Word Encoding

Each word is a 3-tuple `(batch, vector, rule)`:

- **batch** - index into the batch dimension $[0, B)$
- **vector** - index into the activation vector $[0, N)$
- **rule** - grammar rule ID from `TheGrammar` $[0, K)$

Words are stored as a Python list of tuples on the SubSpace.  The list
is a **derivation tree** in pre-order: the first entry is the root, and
binary rules expand left-first.  Terminal words carry `order = -1` and
a codebook index in the `leaf1` slot; the leaf ledger used by
`decompose` is built from these entries.

---

## WordSpace and SymbolicSpace wiring

Under `MentalModel.xml`, a single `WordSpace` hosts one
`syntacticLayer` attribute, built by `_build_syntactic_layer` for the
SymbolicSpace.  The driver methods are:

- `WordSpace.forwardSymbols(data, subspace)` - runs `compose` on the
  SymbolicSpace subspace and returns `(composed, svo)`.
- `WordSpace.reverseSymbols(data, subspace)` - runs `decompose`.
- `WordSpace.predict_rule()` / `predict_rule_hard()` - stack-oriented
  shift/reduce helpers.

`SymbolicSpace.forward` calls `WordSpace.forwardSymbols` as part of
its composition pipeline; the conceptual and perceptual spaces no
longer run their own syntactic layers.

---

## SymbolicSpace

### Forward Path

1. Extract concept activation $[B, n_{\text{Concepts}}]$.
2. Map through `PiLayer(monotonic=True, invertible=True)` to
   $[B, n_{\text{Symbols}}]$.
3. Codebook quantization (when enabled) produces one-hot activation +
   vectors.
4. `WordSpace.forwardSymbols(...)` runs the unified `SyntacticLayer`
   and returns the composed symbolic tensor.

### Reverse Path

1. Extract symbol activation $[B, n_{\text{Symbols}}]$.
2. Exact inverse of the PiLayer recovers $[B, n_{\text{Concepts}}]$.
3. `WordSpace.reverseSymbols(...)` invokes `decompose` to reconstruct
   the pre-compose tensor from the word record.

### Key Properties

- Symbols are **zero-dimensional** -- pure activation scalars, not
  vectors.
- The PiLayer allows $n_{\text{Concepts}} \neq n_{\text{Symbols}}$.
- `composeSyntax()` runs the unified SyntacticLayer and executes the
  soft superposition over all active S-tier rules.

---

## Parse Tree Output

The model emits an XML parse tree via `parse.derivation_to_xml()`.
Word tuples are collected with global Grammar rule IDs and
reconstructed into a tree by recursive descent over the pre-order word
list.

Transition rules (used to close the derivation) are transparent -- they
do not emit XML tags.  Terminal words emit `<token>` leaves.  All other
rules emit their named tag (`<conjunction>`, `<equals>`, `<not>`,
`<union>`, etc.).

Example for "dogs AND NOT cats":

```xml
<conjunction>
  <token word="dogs"/>
  <not>
    <token word="cats"/>
  </not>
</conjunction>
```

---

## AssociationLayer (EQUALS Implementation)

The `AssociationLayer` is a cross-symbol associative memory used by the
EQUALS rule.  Two modes are available:

- **`type="symmetric"`** -- Hopfield-like: learns projection $A$,
  computes association scores $A^\top A$, softmax-retrieves the
  associated pattern.  Associations are symmetric
  ($A \equiv B \Leftrightarrow B \equiv A$).
- **`type="hopfield"`** -- Modern Hopfield: separate query/key
  projections, softmax-gated retrieval.

Input and output are both $[B, N]$ activation vectors.  The layer is
learnable and its parameters are trained end-to-end via the
reconstruction loss.

---

## How Syntax Is Trained

### The Intended Differentiable Mechanism

The syntax code is written to be differentiable:

- `SyntacticLayer.forward()` produces soft `rule_probs`
- `compose()` uses soft superposition over all candidate rules
- `_compose_to_target()` is soft in training, argmax in eval
- The Phase 1 `not` pass is deterministic (non-differentiable) but
  only fires when the top-of-stack mean is negative

A downstream loss can backpropagate into rule probabilities, rule-
prediction weights, depth embeddings, and rule-execution parameters
(e.g. the Sinkhorn logits for `swap`).

### The Actual Losses in `runBatch()`

`BasicModel.runBatch()` computes three losses:

1. output loss
2. optional reconstruction loss
3. optional embedding loss

There is no explicit syntax loss, parse-tree loss, or rule-label
supervision.  Syntax fitness emerges indirectly through the
reconstruction loss: rules that consistently support successful
reconstruction accumulate weight, while weaker alternatives diminish.

### Current Integration Status

Under `MentalModel.xml`:

- syntax is enabled and the unified syntactic layer is instantiated
- syntax rules are predicted, syntax states and word tuples are
  produced
- `MentalModel.forward()` goes through `SigmaLayer` rather than
  invoking the syntax pipeline directly
- syntax is active as computation and analysis, but not strongly
  coupled to the task loss

The syntax system is a real, parameterized, differentiable grammar
module -- configured by XML rule subsets and implemented with a single
`SyntacticLayer` class -- but not yet placed on a strong loss path in
the stock `MentalModel.xml` loop.

---

## Future: RuleNode on the ReconstructionStack

A planned Phase 2 introduces a `RuleNode` structure with slots for
`(rule_id, method_name, forward_op, reverse_op, arity, args)` and
enriches `ReconstructionStack` to support variable-fidelity
reconstruction:

- pos-only
- grammar + pos
- grammar + pos + args (full reconstruction information)

This pairs generative rules with their compositional inverses so that
downstream consumers can pull as much or as little reconstruction
context as they need from the same word record.
