# Language

## Overview

The language system is a soft-superposition CKY chart parser that runs
over symbol activations. `SymbolicSpace` plays the role of the
architecture's **calculator**: the chart at S reads operand cells,
invokes the grammar layer's compose / forward / reverse, and writes the
result back. Three pieces cooperate:

- `Grammar` (in [`bin/Language.py`](../bin/Language.py)) — singleton
  rule table loaded from XML or `data/grammar.cfg`.
- `Chart` (`Language.py`) — owns the chart parameters, runs the inside
  / outside passes, and dispatches each per-cell rule application
  through `wordSpace.host_layer(tier, rule_name)`.
- `GrammarLayer` subclasses (in [`bin/Layers.py`](../bin/Layers.py)) —
  one class per grammar operator (`not`, `non`, `intersection`,
  `union`, `conjunction`, `disjunction`, `lift`, `lower`, `equals`,
  `part`, `true`, `false`, `swap`, `query`).

```
InputSpace  ->  PerceptualSpace  ->  ConceptualSpace  ->  SymbolicSpace  ->  OutputSpace
                                                          chart runs here
```

> **Deferred refactor — serial / shift-reduce parser.** A separate
> follow-up pass will replace the batched CKY with a serial /
> shift-reduce parser driven word-by-word, consuming
> `ConceptualSpace.stm` as the working-memory stack. The cadence is
> **per-word**: each word makes a full round trip
> `P (BPE lex) → C (project) → S (codebook snap = word identity + POS)
> → C (S→C reverse, the unquantized idea) → stm.push(idea)`, and the
> parser then fires reductions on the STM via the chart's grammar
> rules. POS rides the codebook hit for free via the bivector
> codebook's `category_ids` (hard POS tag per atom) and
> `category_logits` (learnable per-atom soft POS distribution); the
> chart's `_apply_codebook_pos_seed` already EMA-updates these from
> the inside pass, so per-word snap returns `(word_id, POS)`
> simultaneously — no separate POS tagger needed.
>
> The chart's launch site is at C (over `ConceptualSpace.stm` snapshot)
> rather than at S (with S retained as a "calculator" reachable via a
> per-reduce dispatch buffer for relational ops like `query` / `equals`
> / `part`). The 2026-05-12 refactor retired the legacy
> `ChartCompose` / `ChartGenerate` stem modules; `_chart_compose_at_C`
> fires inside every body stage and `_chart_generate_from_stm` mirrors
> it on the reverse path. See
> [`doc/plans/2026-05-12-serial-parser-handoff.md`](plans/2026-05-12-serial-parser-handoff.md)
> for the original handoff spec.

### Tiers (P / C / S / L)

| Tier | Owner            | Activation domain                              |
|------|------------------|-------------------------------------------------|
| `P`  | PerceptualSpace  | bivector ``[B, V, 2]`` per percept              |
| `C`  | ConceptualSpace  | bivector ``[B, V, 2]`` pre-codebook             |
| `S`  | SymbolicSpace    | scalar ``[B, V]`` post-codebook (monotonic)     |
| `L`  | (logical, none)  | bivector ``[..., 2]`` -- pure lattice primitive |

S-tier ops compose **monotonic** functions over the post-codebook scalar
activation (``effective_activation()``: bivector poles reduced via
``max`` and gated by modal presence).  L-tier ops (``intersection``,
``union``) are pure lattice min/max on the bivector activation; the
chart binds them at whichever space's tier the operands live in.

Per-space dispatchers are `SyntacticLayer` instances built by
`build_space_syntactic_layer`; each holds a tier-specific
`host_layers` dict keyed by `rule_name`.  Rule-firing probability
gating (`GrammarLayer.gated_run`) lives on `Chart` itself, installed as
`GrammarLayer._chart_authority` at `WordSpace.__init__` time.

---

## Grammar

`TheGrammar` is a singleton `Grammar` instance.  Production configs
load `data/grammar.cfg` through `<WordSpace><language><grammarCfg>...`,
which is the authoritative rule table.  An XML `<grammar>` block is the
legacy alternative.

A `RuleDef` namedtuple carries
`(tier, canonical, arity, method_name, lhs, rhs_symbols)`.
`method_name` is the op name (`'lift'`, `'intersection'`, `'not'`, ...),
or `'merge'` for bare-symbol-sequence rules, or `'emit_head'` for the
downward `S -> C` projection.

### Accessors

- `Grammar.symbolic()` — indices of all S-tier rules
- `Grammar.symbolic_transition()` — `None` (no cross-tier transitions)
- `Grammar.binary_rules()` — indices of all arity-2 rules
- `Grammar.rule_by_id(i)` — canonical production string
- `Grammar.method_name(i)` — dispatch method name (e.g. `'swap'`)
- `Grammar.arity(i)` — 1 or 2
- `Grammar.tier(i)` — `'P'` / `'C'` / `'S'` / `'L'`

### Grammar configuration

`Grammar.configure()` loads an XML `<grammar>` section split into
`<upward>` (parsing), `<downward>` (generation), and `<start>` (the
start symbol).  When the chart reduces a row's parse to a single node
of the start category, the layer emits a soft-reset signal: the outer
doc-streaming loop dispatches `wordSpace.soft_reset(batch=b)`, clearing
per-sentence working buffers.  Discourse history and codebook EMA are
*not* cleared — they are document-scoped.

The loader accepts three RHS forms; the **explicit-op** form is canonical:

```xml
<grammar>
  <upward>
    <!-- (1) Explicit-op form. -->
    <rule>S       = lift(NP, VP)</rule>
    <rule>NP      = intersection(AP, NP)</rule>
    <rule>S       = not(S)</rule>
    <rule>S,S     = intersection_inv(VO)</rule>     <!-- multi-output reverse -->

    <!-- (2) Function-call form. -->
    <S>not(S)</S>
    <S>swap(S, S)</S>

    <!-- (3) Bare-symbol form (transitional). method_name='merge'. -->
    <S>S VP</S>
  </upward>
  <downward>
    <S>C</S>                                        <!-- emit_head -->
  </downward>
</grammar>
```

Op names dispatch into `GRAMMAR_LAYER_CLASSES` (Layers.py): `lift`,
`lower`, `intersection`, `union`, `not`, `non`, `conjunction`,
`disjunction`, `equals`, `part`, `true`, `false`, `swap`, `query`.
Reverse productions are derived mechanically at grammar-load time —
for `LHS = op(arg1, arg2)` the loader synthesizes
`arg1, arg2 = opReverse(LHS)`.

### `data/grammar.cfg` dispatch

Line-oriented text alternative; sections bracketed (`[upward]` /
`[downward]`); each `LHS = body` rule's body is a function call,
a single category (PROJECT), or `epsilon`.  Loading is gated via
`<WordSpace><language><grammarCfg>` in XML.

### Syntactic sugar consumption

Under byte-level lex routed through BPE chunking, whitespace and
punctuation become their own word slots that the grammar reduces away
via `absorb(A, B) -> A`. Initial logits should bias against `absorb`
to avoid early-training collapse.

### Shamatha speech mode

Shamatha speech (`useGrammar="shamathaSpeech"`) is the one-pointed
speech mode.  It may conjoin and disjoin over all active percepts in
the current object field at once, constrained so the logical object
remains single-pointed.  Narrow DNF object grammar:

```ebnf
literal  := polarity? concept
polarity := "" | "non-" | "not"
term     := literal ("and" literal)*
object   := term ("or" term)*
sentence := object
```

Every logical merge must preserve spatiotemporal contiguity:
`where(A)` and `where(B)` must overlap or be connected; `when(A)` and
`when(B)` must overlap or be temporally adjacent.  See
[plans/2026-04-28-shamatha-speech-contiguity-handoff.md](plans/2026-04-28-shamatha-speech-contiguity-handoff.md).

The polarity surface (`""`, `non-`, `not`) maps to the three channels
of `NegationLayer(ternary=True)`.

---

## Operator semantics

Each grammar op has one computation anchor: **Sym** (GrammarLayer
tensor kernel), **SigmaPi** (level-crossing fold via
`SS.forward(CS.forward(...))`), **Def** (definitional WordSpace update),
or **Percepts** (pre-grammar perceptual pre-processing).

### Notation

Let $x, \ell, r \in \mathbb{R}^{B \times N}$ or
$\mathbb{R}^{B \times N \times D}$.  Let $\mathcal{B}$ be the subspace
basis supplying pointwise primitives; absent $\mathcal{B}$, kernels
fall back to torch elementwise.  An optional `mask` gates the output.

### Ternary commitment (`true` / `false` / `non`)

For $x \in [-1, 1]$ (clamped on entry):

$$
\mathrm{true}(x)  = \max(0,\ x), \qquad
\mathrm{false}(x) = \max(0,\ -x), \qquad
\mathrm{non}(x)   = 1 - |x|
$$

These partition unity: $\mathrm{true} + \mathrm{false} + \mathrm{non}
= 1$ pointwise.  All three are lossy.

### Negation (`not`) — Sym

Propositional bivector swap on the leading two dims:
$[x_0, x_1, x_{2..}] \to [x_1, x_0, x_{2..}]$.  Self-inverse.

### Conjunction / disjunction — S-tier (post-codebook scalar) {#conjunction--disjunction--sym}

`S = conjunction(S, S)` and `S = disjunction(S, S)` operate on the
post-codebook scalar activation ``[B, V]`` (non-negative strength per
prototype).  Strictly monotonic lattice primitives:

$$
\mathrm{conjunction}(\ell, r) = \min(\ell, r), \qquad
\mathrm{disjunction}(\ell, r) = \max(\ell, r)
$$

Routed through ``Ops.intersection(..., monotonic=True)`` /
``Ops.union(..., monotonic=True)``.  Both lossy with
``(parent, parent)`` pseudo-inverse on `reverse`.

### Intersection / union — L-tier (lattice on bivectors)

`L = intersection(L, L)` / `L = union(L, L)` are lattice min/max on a
**bivector activation** ``[..., 2]``.  The chart binds them at
whichever space's activation tier the operands live in.

| `monotonic` | kernel                                      |
|-------------|---------------------------------------------|
| `False`     | RadMin / RadMax — same-sign min / max       |
| `True`      | strict lattice min / max (per channel)      |

L-tier ops are lossy: `decompose` returns `(parent, parent)`.

### Part / equals — Def (tensor scoring)

$\mathrm{part}(\ell, r) = \frac{\max(0,\ \ell \cdot r)}{|\ell|\,|r|} \cdot r$.
The scalar score is broadcast to $r$'s rank.  Empty-operand contract:
if $|\ell|$ or $|r|$ is near zero, the score is 1.

`equals(\ell, r) = part(\ell, r) \cdot part(r, \ell)$, scalar-reduced
and broadcast to $r$.  Under `mask`, equals degenerates to L1-distance
agreement.  Both lossy.  See [Mereology.md](Mereology.md).

### Absorb — base-class marker

`absorb(\ell, r) = \ell`.  Lives on `GrammarLayer.absorb(left, right)
-> left` so any binary subclass can flag its right operand as
syntactic sugar.

### Query — Mereological

`query(\ell, r) = \ell`.  The chart pushes a marker word elsewhere
when a norm-drop fires (see [Composition](#composition) below).

### Swap (Sinkhorn soft permutation) — Sym

A learned $3 \times 3$ logit matrix $M$ is normalized via $k=5$
Sinkhorn iterations to produce a doubly-stochastic soft permutation
$P$.  Given operands $\ell, r$ and a learned broadcast marker $m$,
the output is the first row of $P \cdot [\ell, r, m]^\top$.  Lossy.

### Lift / lower — rule-id annotators over the shared subsymbolic loop

Post-2026-05-13 refactor.  `LiftLayer` and `LowerLayer` are now
**pure rule-id annotators** — they no longer own (or borrow) a
substrate sigma/pi.  All composition happens in the unconditional
subsymbolic loop:

```
C  =  sigma_percept(  pi_input(IS)  +  pi_concept(C_prev)  )
```

owned by `ConceptualSpace.sigma_percept` and the
`PerceptualSpace.pi_input` / `pi_concept` pair.  See
[Spaces.md §"Sigma / Pi
ownership"](Spaces.md#sigma--pi-ownership-2026-05-13-rebalance).

When the chart fires `S = lift(NP, VP)` or `S = lower(NP, VP)`,
`LiftLayer` / `LowerLayer`:

1. Record the **rule_id** in the parse tree (`lift` vs `lower`).
2. Optionally stamp the per-slot **catuskoti tag** on
   `STM._truth_tags` so downstream readers (truth layer, output
   decoder) can read role in O(1).

They do **not** apply a separate sigma / pi.  The composed C-state is
whatever the unconditional loop already produced this tick.  Lift vs
lower differ in what the downstream **truth and output layers** make
of that state, not in the substrate operation that produced it.

#### Why no separate substrate per rule

Linguistically, *"the running boy"* (lowering, attribution) and
*"the boy runs"* (lifting, predication) share a single neural
composition act — *fuse a noun representation with a verb
representation into one bound state*.  The lift/lower distinction is a
**derivational labelling** of that shared state, not a separate
operation.  Re-firing the loop with a different rule annotation
suffices to switch framings; the chart records which framing won.

The previous "gated-substrate" pattern (which borrowed
`perceptualSpace.sigma` at concept_dim and
`conceptualSpace.pi` at percept_dim) is retired.  Standalone-test
construction with no perceptual/conceptual refs falls back to the
static lattice kernels (`Ops._lift_kernel` / `Ops._lower_kernel`)
when the unconditional loop isn't wired (legacy compatibility).

#### Idempotence of the symbolic loop

`cs.forward(ss.forward(c)) == c`.  SymbolicSpace is a dimensional
pass-through (no default sigma/pi at S, only grammar ops, which are
idempotent under their algebra), so routing a C-activation through SS
and back through CS returns the same state.  An unconditional
`pi_concept` therefore can't double-apply across the symbolic round
trip — it just re-folds the same C content into the unchanged P
state.

### Chunking — Percepts

Not a grammar rule.  Implemented by `ChunkLayer` /
`PerceptualSpace.chunk_static` as a perceptual pre-pass before the
grammar sees symbol slots.

### Mereological suite — see [Mereology.md](Mereology.md)

`part`, `whole`, `equal`, `overlap`, `underlap`, `boundary`, `copart`
live in `Ops.*`; the canonical spec is [Mereology.md](Mereology.md).
Each has a vector form (default) and a scalar form (`scalar=True`).

Empty-set conventions (parthood scalar form):
$\mathrm{part}(\emptyset, y) = 1$, $\mathrm{part}(x, \emptyset) = 0$,
$\mathrm{part}(\emptyset, \emptyset) = 1$.

---

## GrammarLayer Implementations

`GrammarLayer` (in `bin/Layers.py`) is the base class.  Each subclass
declares `rule_name`, `arity`, `invertible`, `lossy` as class attributes,
then overrides `forward()` (and `reverse()` if `invertible`).  Binary
subclasses additionally override `compose(left, right)` and
`generate(parent)`.

### Base contract

Class attributes: `rule_name`, `arity`, `invertible`, `lossy`,
`_chart_authority = None`.

`gated_run(x, fn, ...)` consults `_chart_authority.should_run_rule(rule_name)`
to get firing probability `p`, then returns `fn(x)` if `p >= 1`, `x` if
`p <= 0`, or the soft mixture `p * fn(x) + (1 - p) * x` otherwise.

`compose(*operands)` defaults to `forward(operands[0])` for arity 1;
arity 2 raises `NotImplementedError` and must be overridden.  `generate(parent)`
defaults to `reverse(parent)` for arity 1 invertible, else `parent`.

### Layer summary

| Class | Rule | Tier | Arity | Reads | Description |
|-------|------|------|-------|-------|-------------|
| `NotLayer` | `not` | S | 1 | event | Bivector pole swap on `[..., :2]`; self-inverse |
| `NonLayer` | `non` | S | 1 | event | Per-pole bivector complement `1 - x[..., :2]`; self-inverse |
| `IntersectionLayer` | `intersection` | L | 2 | activation | Lattice min on bivector `[..., 2]`; `monotonic` toggles RadMin / strict |
| `UnionLayer` | `union` | L | 2 | activation | Lattice max on bivector `[..., 2]`; `monotonic` toggles RadMax / strict |
| `ConjunctionLayer` | `conjunction` | S | 2 | activation | `Ops.intersection(monotonic=True)` on `[B, V]` scalar |
| `DisjunctionLayer` | `disjunction` | S | 2 | activation | `Ops.union(monotonic=True)` on `[B, V]` scalar |
| `LiftLayer` | `lift` | S | 2 | event | `sigma.compose` (OR fold) gated by per-rule `raw_gate`; exact LDU reverse |
| `LowerLayer` | `lower` | S | 2 | event | `pi.compose` (AND fold) gated by per-rule `raw_gate`; exact LDU reverse |
| `EqualsLayer` | `equals` | S | 2 | event | Mutual parthood score broadcast over right operand; lossy |
| `PartLayer` | `part` | S | 2 | event | Directional parthood score broadcast over right operand; lossy |
| `TrueLayer` | `true` | S | 1 | event | `relu(clamp(x, -1, 1))`; lossy |
| `FalseLayer` | `false` | S | 1 | event | `relu(-clamp(x, -1, 1))`; lossy |
| `SwapLayer` | `swap` | S | 2 | event | Sinkhorn soft permutation; `swap_logits`+`swap_marker` parameters |
| `QueryLayer` | `query` | S | 2 | event | Returns left operand; chart pushes marker elsewhere |

`reads_activation = True` feeds `subspace.materialize(mode='activation')`;
`False` feeds the muxed event tensor.

### `NotLayer` example

```python
class NotLayer(GrammarLayer):
    rule_name        = "not"
    arity            = 1
    invertible       = True
    tier             = 'S'
    reads_activation = False

    def forward(self, x):
        bivector = x[..., :2].flip(dims=(-1,))
        rest     = x[..., 2:]
        return torch.cat([bivector, rest], dim=-1) if rest.shape[-1] else bivector

    def reverse(self, y):
        return self.forward(y)
```

### `IntersectionLayer` / `UnionLayer` example

```python
class IntersectionLayer(GrammarLayer):
    rule_name        = "intersection"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'L'
    reads_activation = True

    def __init__(self, monotonic=False):
        super().__init__(0, 0)
        self.monotonic = bool(monotonic)

    def forward(self, left, right):
        return Ops.intersection(left, right, monotonic=self.monotonic)

    def reverse(self, parent):
        return parent, parent
```

`UnionLayer` is identical with `Ops.union` and `rule_name = "union"`.

### `PiLayer` / `SigmaLayer` — parametrized folds

`PiLayer` (AND fold) and `SigmaLayer` (OR fold) inherit `ButterflyLayer`.
Unary `forward(x)` is the log-domain multiplicative fold over the feature
axis; binary `compose(left, right)` sums log-mult contributions across
operands instead of within one.

PiLayer's compose: `(log(_to_mult(left)) + log(_to_mult(right))) @ W +
bias` followed by `tanh(.../2)`.  Generate inverts via
`compute_Winverse_current()`.  SigmaLayer's compose is the additive
(logit-domain) analog: atanh the operands, sum across operands, linear,
tanh.  `binary=True` on either pre-applies `Ops.top2_select_ste` to
hard-select the top-2 operands.

### Class registry

```python
GRAMMAR_LAYER_CLASSES = {
    # S-tier (post-codebook scalar / muxed event)
    'not':          NotLayer,
    'non':          NonLayer,
    'conjunction':  ConjunctionLayer,
    'disjunction':  DisjunctionLayer,
    'lift':         LiftLayer,
    'lower':        LowerLayer,
    'equals':       EqualsLayer,
    'part':         PartLayer,
    'true':         TrueLayer,
    'false':        FalseLayer,
    'swap':         SwapLayer,
    'query':        QueryLayer,
    # L-tier (lattice on bivector activation)
    'intersection': IntersectionLayer,
    'union':        UnionLayer,
}
```

### Inverse contract by operator

| Op            | Tier | Reverse                                        |
|---------------|------|-------------------------------------------------|
| `not`         | S    | exact (self-inverse pole flip)                 |
| `non`         | S    | exact (self-inverse)                           |
| `lift`        | S    | exact LDU inverse via gated ``sigma.generate`` |
| `lower`       | S    | exact LDU inverse via gated ``pi.generate``    |
| `conjunction` | S    | pseudo-inverse ``(parent, parent)``            |
| `disjunction` | S    | pseudo-inverse ``(parent, parent)``            |
| `intersection`| L    | pseudo-inverse ``(parent, parent)``            |
| `union`       | L    | pseudo-inverse ``(parent, parent)``            |
| `equals`      | S    | no defined inverse (mereological write)        |
| `part`        | S    | no defined inverse (mereological write)        |
| `query`       | S    | no defined inverse (mereological read)         |
| `swap`        | S    | pseudo-inverse ``(parent, parent)``            |
| `true`        | S    | identity passthrough (lossy)                   |
| `false`       | S    | identity passthrough (lossy)                   |

---

## Rule prediction

`SyntacticLayer.predict_rules(x)` — also on `Chart` — predicts a rule
distribution at each derivation depth.  The architecture is weight-tied
across depths with a learned depth embedding.

Input projection:

$$
h^{(0)} = \sigma(W_{\text{in}}\, x + b_{\text{in}}),
\qquad h^{(0)} \in \mathbb{R}^{B \times d_{\text{hidden}}}
$$

For each depth $d = 0, 1, \ldots, D_{\max} - 1$ (shared weights):

$$
\begin{aligned}
h^{(d+1)} &= \sigma\!\bigl(W_{\text{deriv}}\,(h^{(d)} + e_d) + b_{\text{deriv}}\bigr) \\
z^{(d)}   &= W_{\text{head}}\,h^{(d+1)} + b_{\text{head}} \in \mathbb{R}^{B \times K} \\
p^{(d)}   &= \begin{cases}
              \mathrm{gumbel\_softmax}(z^{(d)}; \tau) & \text{training} \\
              \mathrm{softmax}(z^{(d)})               & \text{eval}
             \end{cases}
\end{aligned}
$$

Output: `{rule_logits, rule_probs, predicted_rules}`.  Training uses
gumbel-softmax; eval uses softmax + argmax.

---

## Composition

Chart composition lives on `Chart` (`bin/Language.py`).
`Chart.compose(data, word_space, subspace=None)` runs the inside pass
and populates per-tier rule selections on `word_space.current_rules`.
`Chart.generate(target, word_space, subspace=None)` runs the outside
pass + Viterbi backtrace.

Router selected via `<WordSpace><routerKind>`: `'chart'` (CKY,
default) or `'signal'` (tensorial alternative).

### Chart inside pass

For each cell $(i, j)$ and each grammar rule with matching arity:

$$
\mathrm{score}_{ij}^{\rho} = \mathrm{compat}(\text{lhs}, \text{rhs}_{ij}) \cdot
                              \mathrm{pair\_score}(i, j) \cdot
                              \mathrm{rule\_logit}(\rho)
$$

Soft mixture (training) or argmax (eval).  Per-cell applications
dispatch through `wordSpace.host_layer(tier, rule_name)`.  The
compatibility mask `compat[B, P, R]` zeroes `(pair, rule)` combinations
whose typed LHS / RHS categories don't match the pair's slot
categories.  Category `0` (`'?'`) is wildcard.

#### Query / norm-drop detection

Detect symbolic contradiction at the accumulation point by comparing
pre- and post-composition Frobenius norms.  Define $\nu_\ell =
\|\ell\|_F$ and $\nu_c = \|c\|_F$.  When $\nu_\ell > 10^{-6}$ and
$\nu_c < 0.1\,\nu_\ell$ (`_QUERY_NORM_DROP_RATIO = 0.1`), the
accumulator preserves the prior state and a `query` word is pushed.

### Tensor word buffer

Each `SubSpace` carries two registered buffers alongside the legacy
`self.word: list[tuple]`:

```python
self.word_records  # [B*K, max_depth, ENTRY_WIDTH] long, ENTRY_WIDTH=7
self.word_count    # [B*K] long, current depth per cell
```

`add_word(...)` is overloaded.  The chart materializes its tensor
buffer before `_compose_chart_cky` returns via
`subspace.flush_word_buffer()`.

### Chart-derived SVO

`Chart._extract_svo_from_trace` walks each batch row's trace:

1. Find the last `S -> S VO` entry (outermost reduction).
2. Find the matching `VO -> V O` entry whose `merged_slot` equals the
   outer rule's right arg.
3. Pull `subject` from the outer's left slot, `verb` from the inner's
   left, `object` from the inner's right.

---

## POS side-channel

A part-of-speech distribution rides alongside the signal at every
chart cell.  Trailing axis is `|POS| = |Grammar.categories|`.  Index 0
is `'?'`, the wildcard.

### Storage tiers

- **Per-atom POS tag** (durable seed): `Codebook.category_ids: Long[V]`
  on the SymbolicSpace codebook.  Use `set_category(idx, cat_id)` /
  `get_category(idx)`.
- **Per-call chart POS**: `Chart._chart_pos: [B, N+1, N+1, |POS|]`,
  softmax of `_chart_score`. Cleared at start of each inside pass.
- **Durable per-merge POS**: `SubSpace.pos_records: Long[B*K,
  max_depth]`, parallel to `word_records` / `word_count`.

### Lexical bootstrap

`Chart._apply_codebook_pos_seed` overrides the learned
`_lex_cat_scorer(data)` log-distribution with a one-hot at the
codebook-resolved POS for any input position whose nearest atom
carries a non-wildcard `category_ids` entry and whose dot-product
similarity exceeds `0.1`.

### Mechanism 1 — RHS POS compatibility mask

Before scoring each binary merge candidate, the chart gathers operand
POS distributions and adds `log(p_left * p_right)` to the rule's
candidate score.  Wildcard rules (`rhs_*[r] == 0`) get an unconditional
`1.0` multiplier.

### Mechanism 3 — Rule-prediction conditioning

After Viterbi extraction, `Chart._populate_category_stack` walks the
trace and pushes each merge's LHS category embedding onto
`WordSpace.category_stack`.  Subsequent `WordSpace.predict_rule(b)`
reads a parse history of POS embeddings via the existing
`Linear(max_depth * pos_dim -> n_rules)` MLP.

### Why this trains POS to every word

Two reinforcing gradient paths:

1. **Anchored** — codebook atoms with a known category override
   `_lex_cat_scorer` deterministically; reconstruction loss pulls
   downstream rules toward those categories.
2. **Emergent** — untagged atoms fall back to
   `softmax(_lex_cat_scorer(data))`; a wrong POS at a leaf produces a
   wrong rule firing, bad reconstruction, and a strong gradient back
   through `_lex_cat_scorer.W`.

Coverage matters — tagging closed-class words (DET, AUX, PREP, CONJ)
plus frequent open-class atoms gives a working anchor set.

### `<writeSyntax>true</writeSyntax>` — syntax-tree dump

When set, `BasicModel.forward()` calls `write_syntax_tree(path)` at
the end of every forward pass.  Default path `output/syntax.xml`
(overridable via `<architecture><syntaxOutPath>`).

The dumper walks `_derivation_trace` and emits one `<forward>`
fragment per call:

```xml
<forward tick="42">
  <batch n="0">
    <node cat="S" rule="S=conjunction(S,S)" i="0" j="2">
      <leaf token="1" pos="S" i="0"/>
      <leaf token="1" pos="S" i="1"/>
    </node>
    <rules>1</rules>
  </batch>
</forward>
```

`<node>` carries `cat`, `rule`, `i`/`j`; `<leaf>` carries `token`,
`pos`, input position `i`.  `<rules>` lists the global rule-id
sequence for diff-based regression.  File truncated on first call,
appended on subsequent calls.

Under `<routerKind>signal</routerKind>`, no derivation trace is
recorded; the dumper emits `<noTrace/>` for those rows.

---

## Per-space dispatch (`SyntacticLayer`)

Each Space carries a `SyntacticLayer` instance whose `forward()` /
`reverse()` fires one fold step per call, per the chart's per-tier
rule selection.  `build_space_syntactic_layer` constructs:

- `host_layers`: dict mapping `rule_name` to a `GrammarLayer` instance
  (parametrized folds like `space.pi`, `space.sigma`, plus stateless
  rules built lazily from `GRAMMAR_LAYER_CLASSES`).
- Each layer is registered with the WordSpace via
  `register_host_layer(tier, rule_name, layer)`.

`SyntacticLayer.forward(subspace)` reads
`word_space.current_rules[tier]`, advances a per-tier cursor, and
dispatches to the chosen layer's `forward()` (arity-1 rules only;
binary rules fire inside the chart's `compose`).  `reverse()` mirrors
via `word_space.generate_rules`.

### `WordSpace.forwardSymbols` / `reverseSymbols`

Thin: `forwardSymbols(data, subspace)` demuxes the muxed symbol tensor
into the subspace's modality slots when shapes match.  The symbolic
composition runs on the chart via `ChartCompose`.  `reverseSymbols` is
pass-through.

---

## Downward head emission (`S -> C`)

Selected by `<WordSpace><downwardGeneration>true</downwardGeneration>`.
After the upward parse reduces N leaves to a single root vector,
`BasicModel.forward` takes that root and calls
`WordSpace.reconstruct(state, inputSpace)`, which runs the downward
`S -> C` rule as a one-shot projection onto a codebook.

### Projection-coefficient scoring

$$
W_{\text{norm}} = W / \lVert W\rVert_{2,\text{row-wise}},
\qquad
\text{scores} = \mathrm{clamp}_{\ge 0}(\text{state} \cdot W_{\text{norm}}^\top)
$$

The per-row argmax is `best_idx`.  `state` is *not* L2-normalized
(only the codebook rows are), so the score is "how much of atom $k$
lives in the state".  `contained = scores[:, k] * W_{\text{norm}}[k]`
is the slice of the atom actually contained in the state;
`residual = state - contained` is the leftover.

### `WordSpace.reconstruct(state, codebook_space)`

Any space whose `.subspace.basis` exposes a `getW() -> [V, D]` works.
`InputSpace` (word embedding) is the intended source.  Returns
`{heads, contained, residual, state}`.  `BasicModel.forward` stashes
`heads` as `self._predicted_head`.

---

## Word encoding

Each word is a 7-tuple `(batch, vector, rule, leaf1, leaf2, leaf3,
order)` stored in `subspace.word` (legacy list) and `word_records`
(tensor buffer).  Terminal words carry `order = -1` and a codebook
index in `leaf1`.  The list is a derivation tree in pre-order: root
first, binary rules expand left-first.

---

## Decomposition

`Chart.generate(target, word_space, subspace)` reconstructs the
pre-compose tensor from the symbolic word record by walking the
outside pass + Viterbi backtrace.  Dispatches `host_layer.generate(parent)`
for binary rules, `host_layer.reverse(y)` for unary invertible rules.
Lossy rules emit pseudo-inverses.

Codebook fast path: if a basis is available and shapes match, the
generator looks up each terminal word (those with `order = -1`) and
places `cb[leaf1]` at the recorded position.

---

## How syntax is trained

The syntax code is end-to-end differentiable: soft `rule_probs`, soft
superposition over compatible rules, gradients flow through fold
weights (PiLayer / SigmaLayer / SwapLayer logits).

`BasicModel.runBatch()` computes three losses: output, optional
reconstruction, optional embedding.  No explicit syntax loss; syntax
fitness emerges indirectly through reconstruction — rules supporting
successful reconstruction accumulate weight.

---

## Two-Layer logic

**Subsymbolic** layer treats objects as vector sets $(B, N, D)$ as
RBF / luminosity fields: union is $\max(\ell, r)$; intersection is
$\min(\ell, r)$; negation is $-x$; non is $\alpha x$ with $\alpha \in
[0, 1)$; parthood is fuzzy max-coverage in $[-1, 1]$.

A **symbolization** map projects vectors to scalar truth strength:

$$
s(X) = 2 \cdot \mathrm{mean}(\|x_i\|) - 1 \quad \in [-1, 1]
$$

The **symbolic** layer operates on scalars in $[-1, 1]$: neg is $-a$,
non is $\alpha a$, union is $\max(a, b)$, intersection is $\min(a, b)$,
part is $\mathrm{clamp}(b - a, -1, 1)$.

Subsymbolic = geometry; symbolic = order + polarity; symbolization =
norm projection.

---

## SymbolicSpace integration

### Forward path

1. Extract concept activation $[B, n_{\text{Concepts}}]$.
2. Map through `PiLayer(monotonic=True, invertible=True)` to
   $[B, n_{\text{Symbols}}]$.
3. Codebook quantization produces one-hot activation plus vectors.
4. The chart runs the inside pass over symbol vectors via
   `wordSpace.host_layer('S', rule_name)`.

### Reverse path

1. Extract symbol activation $[B, n_{\text{Symbols}}]$.
2. Exact inverse of the PiLayer recovers $[B, n_{\text{Concepts}}]$.
3. The chart's outside pass reconstructs the pre-compose tensor via
   `host_layer('S', rule_name).generate(parent)`.

### Key properties

- Symbols are zero-dimensional — pure activation scalars.
- The PiLayer allows $n_{\text{Concepts}} \neq n_{\text{Symbols}}$.
- Rule firing gated by LHS / RHS POS compatibility via `chart_pos`.

---

## AssociationLayer (EQUALS implementation)

Cross-symbol associative memory used by EQUALS.  Two modes:
**`type="symmetric"`** Hopfield-like (learns $A$, computes $A^\top A$,
softmax-retrieves; symmetric associations); **`type="hopfield"`**
modern Hopfield (separate query / key projections + softmax-gated
retrieval).  Input and output both $[B, N]$.  Trained end-to-end via
reconstruction loss.
