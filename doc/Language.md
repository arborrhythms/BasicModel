# Language

## Overview

The language system is a soft-superposition CKY chart parser that runs
over symbol activations.  Three pieces cooperate:

- `Grammar` (in [`bin/Language.py`](../bin/Language.py)) — singleton
  rule table loaded from XML or `data/grammar.cfg`.
- `Chart` (`Language.py`) — owns the chart parameters, runs the inside
  / outside passes, and dispatches each per-cell rule application
  through `wordSpace.host_layer(tier, rule_name)`.
- `GrammarLayer` subclasses (in [`bin/Layers.py`](../bin/Layers.py)) —
  one class per grammar operator (`not`, `non`, `intersection`,
  `union`, `conjunction`, `disjunction`, `lift`, `lower`, `equals`,
  `part`, `true`, `false`, `swap`, `query`).  Each class exposes
  `forward()` / `reverse()` (the unary `Layer` interface) and, for
  binary ops, `compose()` / `generate()` (the chart's binary tensor
  interface).

```
InputSpace  ->  PerceptualSpace  ->  ConceptualSpace  ->  SymbolicSpace  ->  OutputSpace
                                                          chart runs here
```

### Tiers (P / C / S / L)

Each rule and each layer carries a tier tag.  The four tiers map to
distinct activation domains:

| Tier | Owner            | Activation domain                              |
|------|------------------|-------------------------------------------------|
| `P`  | PerceptualSpace  | bivector ``[B, V, 2]`` per percept              |
| `C`  | ConceptualSpace  | bivector ``[B, V, 2]`` pre-codebook             |
| `S`  | SymbolicSpace    | scalar ``[B, V]`` post-codebook (monotonic)     |
| `L`  | (logical, none)  | bivector ``[..., 2]`` -- pure lattice primitive |

S-tier ops compose **monotonic** functions over the post-codebook
scalar activation (``effective_activation()``: bivector poles reduced
via ``max`` and gated by modal presence).  L-tier ops
(``intersection``, ``union``) are pure lattice min/max on the bivector
activation; they aren't owned by any space.  The chart binds an
L-tier op at whichever space's tier the operands live in
(``intersection(C, C)`` binds at C; the L tag is layer-side
classification, not a routing tier).

History: the 2026-04-19 merge collapsed the C-tier into S and removed
the P-tier syntactic layer (chunking became a non-syntactic pre-pass).
The 2026-05-01 refactor moved per-rule math out of `SyntacticLayer`
onto `GRAMMAR_LAYER_CLASSES`.  The 2026-05-04 cleanup retired the
``Fusion`` / ``Contiguous`` / ``what`` / ``where`` / ``when`` /
``absorb`` operators (Fusion duplicated DisjunctionLayer; the slot
selectors were dispatcher concerns; absorb became a base-class
sentence marker).  The 2026-05-05 directive split lattice-min/max
into the new ``L`` tier and pinned S-tier conjunction/disjunction to
the post-codebook scalar activation domain.

Per-space dispatchers are `SyntacticLayer` instances built by
`build_space_syntactic_layer`; each holds a tier-specific
`host_layers` dict keyed by `rule_name`.  The chart authority role
(rule-firing probability gating via `GrammarLayer.gated_run`) lives
on `Chart` itself — `WordSpace.__init__` installs the chart as
`GrammarLayer._chart_authority` and hands it the `Grammar` reference;
the legacy module-global `SyntacticLayer` that previously held this
responsibility was retired 2026-05-08.  The dispatcher's
``_read_subspace`` / ``_write_subspace`` honor each layer's
``reads_activation`` flag: ``True`` reads/writes the activation
field (bivector at C-tier, scalar at S-tier), ``False`` reads/writes
the muxed event tensor.

---

## Grammar

`TheGrammar` is a singleton `Grammar` instance.  Production configs
load `data/grammar.cfg` through `<WordSpace><language><grammarCfg>...`,
which is the authoritative rule table.  An XML `<grammar>` block
inside `<WordSpace><language>` is the legacy alternative; both paths
populate the same `Grammar.rules` list.

A `RuleDef` namedtuple carries
`(tier, canonical, arity, method_name, lhs, rhs_symbols)`.
`method_name` is the op name from the explicit-op / function-call RHS
(`'lift'`, `'intersection'`, `'not'`, ...), or `'merge'` for
bare-symbol-sequence rules, or `'emit_head'` for the downward
`S -> C` projection.

### Accessors

- `Grammar.symbolic()` — indices of all S-tier rules
- `Grammar.symbolic_transition()` — `None` (no cross-tier transitions)
- `Grammar.binary_rules()` — indices of all arity-2 rules
- `Grammar.rule_by_id(i)` — canonical production string
- `Grammar.method_name(i)` — dispatch method name (e.g. `'swap'`)
- `Grammar.arity(i)` — 1 or 2
- `Grammar.tier(i)` — `'P'` / `'C'` / `'S'` / `'L'`

### Grammar configuration

`Grammar.configure()` loads an XML `<grammar>` section.  The block
splits into `<upward>` (parsing / chart compose), `<downward>`
(generation from a deep symbolic state), and `<start>` (the start
symbol).

The `<start>S</start>` element is the canonical sentence boundary
marker.  When the chart reduces a row's parse to a single node of the
start category, the layer emits a host-side soft-reset signal for
that row.  The outer doc-streaming loop drains the signal each tick
and dispatches `wordSpace.soft_reset(batch=b)`, which clears every
per-sentence working buffer for row `b` (parse stack, category stack,
reconstruction stack, `_last_svo[b]`, `_stm_fired[b]`).  Discourse
history (`InterSentenceLayer` ring buffer) and codebook EMA are *not*
cleared — they are document-scoped.  Hard reset (document boundary)
wipes both.

The loader accepts three RHS forms; the **explicit-op** form is the
canonical target:

```xml
<grammar>
  <upward>
    <!-- (1) Explicit-op form. RHS is the dispatched op call;       -->
    <!--     LHS may be a comma-separated tuple for multi-output    -->
    <!--     reverses.                                                -->
    <rule>S       = lift(NP, VP)</rule>
    <rule>NP      = intersection(AP, NP)</rule>
    <rule>S       = not(S)</rule>
    <rule>S,S     = intersection_inv(VO)</rule>     <!-- multi-output reverse -->

    <!-- (2) Function-call form. Tag = LHS, body = op(args).         -->
    <S>not(S)</S>
    <S>swap(S, S)</S>
    <S>absorb(S, S)</S>

    <!-- (3) Bare-symbol form (transitional). Dispatches via         -->
    <!--     method_name='merge'.                                     -->
    <S>S VP</S>
    <NP>AP NP</NP>
  </upward>
  <downward>
    <!-- One-shot head emission via method_name='emit_head'.         -->
    <S>C</S>
  </downward>
</grammar>
```

The op names in the explicit-op RHS are dispatch keys into
`GRAMMAR_LAYER_CLASSES` (Layers.py); `lift`, `lower`, `intersection`,
`union`, `not`, `non`, `conjunction`, `disjunction`, `equals`, `part`,
`true`, `false`, `swap`, `query`.  Reverse productions are *derived
mechanically* at grammar-load time — for each forward rule
`LHS = op(arg1, arg2)` the loader synthesizes
`arg1, arg2 = opReverse(LHS)` (multi-return).

Retired ops (do not use in new grammars):
- `Fusion` / `Contiguous` — duplicate of `disjunction` at S-tier;
  migrate to `disjunction(S, S)`.
- `what` / `where` / `when` — slot partitioning is the dispatcher's
  responsibility, not a grammar rule.
- `absorb` — sentence-marker behavior moved to
  `GrammarLayer.absorb(left, right) -> left` on the base class.

### `data/grammar.cfg` dispatch

The text-based rule table is line-oriented:

```
# comments start with '#'
[upward]
S = NP                        # PROJECT (single-category RHS)
S = lift(NP, VP)
NP = intersection(AP, NP)
S = not(S)
S = swap(S, S)
S = absorb(S, S)
...

[downward]
C = emit_head(S)
```

Sections are bracketed (`[upward]` / `[downward]`).  Each `LHS = body`
rule's body is either a function call `op(arg1, arg2)`, a single
category (PROJECT), or `epsilon`.  Loading is gated via
`<WordSpace><language><grammarCfg>` in the XML config — when set,
`Grammar._ensure_configured()` calls `Grammar.load_from_cfg(...)` and
skips the legacy `<grammar>` block.

### Syntactic sugar consumption

Under the byte-level lex routed through BPE chunking, raw text
positions include syntactic sugar — runs of whitespace, single
punctuation marks, newlines.  After chunking these become their own
word slots that the grammar must reduce away.  The post-rolling-cursor
grammar adds one explicit absorption rule: `absorb(A, B)` returns `A`
and discards `B`.  Initial logits should bias against `absorb` to
avoid early-training collapse where the parser learns to discard
everything past the first token.

### Shamatha speech mode

Shamatha speech is the one-pointed speech mode, configurable via
`useGrammar="shamathaSpeech"` (with `thoughtFree` kept as a
compatibility alias).  It is *not* serial mode; serial mode constrains
the runtime cursor to a moving prefix / next-token task.  Shamatha
speech may conjoin and disjoin over all active percepts in the
current object field at once — its constraint is that the logical
object remains single-pointed.

The narrow grammar is the DNF object grammar:

```ebnf
literal  := polarity? concept
polarity := "" | "non-" | "not"
term     := literal ("and" literal)*
object   := term ("or" term)*
sentence := object
```

For this grammar to express a single object rather than a scattered
aggregate, every logical merge must preserve spatiotemporal
contiguity: `where(A)` and `where(B)` must overlap, touch, or be
connected by an allowed adjacency edge before any merge may form a
larger object; `when(A)` and `when(B)` must overlap or be temporally
adjacent.  See
[plans/2026-04-28-shamatha-speech-contiguity-handoff.md](plans/2026-04-28-shamatha-speech-contiguity-handoff.md).

The polarity surface (`""`, `non-`, `not`) maps to the three channels
of `NegationLayer(ternary=True)`: affirmation `x`, non-affirming
negation `non(x)`, and affirming negation `-x`.

---

## Operator semantics

Every grammar op has one explicit computation anchor: **Sym** (call a
GrammarLayer's tensor kernel directly), **SigmaPi** (level-crossing
fold via `SS.forward(CS.forward(...))` and its inverse), **Def**
(definitional WordSpace update over the symbolic codebook /
mereological graph), or **Percepts** (perceptual pre-processing —
chunking, before the grammar sees symbol slots).

### Notation

Let $x, \ell, r \in \mathbb{R}^{B \times N}$ (activation mode) or
$\mathbb{R}^{B \times N \times D}$ (vector mode).  Let $\mathcal{B}$ be
the subspace basis (a bitonic `Basis` or codebook), supplying
pointwise primitives `pos`, `negation`, `conjunction`, `disjunction`,
`equal`, `part`.  When $\mathcal{B}$ is absent the kernels fall back
to torch elementwise primitives.  An optional concept-axis `mask`
gates the output by zeroing non-selected dimensions.

### Ternary commitment ops (`true` / `false` / `non`)

For $x \in [-1, 1]$ (clamped on entry):

$$
\mathrm{true}(x)  = \max(0,\ x), \qquad
\mathrm{false}(x) = \max(0,\ -x), \qquad
\mathrm{non}(x)   = 1 - |x|
$$

These three partition unity: $\mathrm{true} + \mathrm{false} +
\mathrm{non} = 1$ pointwise.  The ReLU shape of `true` / `false` makes
them the "committed yes" and "committed no" halves of a bitonic axis;
`non` is the triangular indeterminate residual peaked at $x = 0$.
All three are lossy.

### Negation (`not`) — Sym

`not` is the propositional bivector swap on the leading two dims of a
symbol vector: $[x_0, x_1, x_{2..}] \to [x_1, x_0, x_{2..}]$.  Pure
antipodal flip on the bitonic axis; self-inverse.

### Conjunction / disjunction — S-tier (post-codebook scalar) {#conjunction--disjunction--sym}

`S = conjunction(S, S)` and `S = disjunction(S, S)` operate on the
**post-codebook scalar activation**: a ``[B, V]`` tensor where ``V``
indexes prototypes in the symbolic codebook and each entry is the
non-negative strength of that prototype after the codebook snap.
Concretely, ``materialize(mode='activation')`` returns
``effective_activation()`` -- the bivector poles ``[pos, neg]``
reduced via ``max(pos, neg)`` and gated by modal presence.

Because the activation is non-negative scalar, both ops are
**strictly monotonic** lattice primitives:

$$
\mathrm{conjunction}(\ell, r) = \min(\ell, r), \qquad
\mathrm{disjunction}(\ell, r) = \max(\ell, r)
$$

(routed through ``Ops.intersection(..., monotonic=True)`` and
``Ops.union(..., monotonic=True)``, which collapse to ``torch.min``
/ ``torch.max`` via ``_lower_kernel(kind='strict')`` /
``_lift_kernel(kind='strict')``).  Both are lossy with
``(parent, parent)`` pseudo-inverse on `reverse`.

### Intersection / union — L-tier (lattice on bivectors)

`L = intersection(L, L)` and `L = union(L, L)` are pure logical
primitives: lattice min/max on a **bivector activation** ``[..., 2]``.
The L tier means the layer isn't owned by any single space -- the
chart binds it to whichever space's activation tier the operands
live in.  In practice the upstream tier is C (where activation is a
``[B, V, 2]`` bivector pre-codebook); the same kernel works at S
when the chart sources S-tier bivector activation, but S-tier
production grammars typically use ``conjunction`` / ``disjunction``
on the post-codebook scalar instead.

The kernels accept a ``monotonic`` kwarg:

| `monotonic` | kernel                                      |
|-------------|---------------------------------------------|
| `False`     | RadMin / RadMax — same-sign min / max       |
|             | magnitude with zero passthrough             |
| `True`      | strict lattice min / max (per channel)      |

L-tier ops are lossy: `decompose` returns `(parent, parent)` as the
pseudo-inverse.  When the upstream Basis carries codebook
weights, callers can substitute ``Ops.lowerReverseAll(Y, W,
monotonic)`` / ``Ops.liftReverseAll(Y, W, monotonic)`` for a
codebook-search recovery instead of the identity pseudo-inverse.

### Part / equals — Def (currently tensor scoring)

Parthood weights the right operand by its containment in the left:
$\mathrm{part}(\ell, r) = \frac{\max(0,\ \ell \cdot r)}{|\ell|\,|r|} \cdot r$.
The scalar score is broadcast to $r$'s rank.  Empty-operand contract:
if $|\ell|$ or $|r|$ is near zero, the score is 1 (the empty set is
part of everything).

`equals(\ell, r)` is mutual parthood:
$\mathrm{part}(\ell, r) \cdot \mathrm{part}(r, \ell)$, scalar-reduced
and broadcast to $r$.  Under a `mask`, equals degenerates to an
L1-distance agreement score on the selected dims.  Both are lossy.
The target architecture moves both from tensor scoring to graph
updates in WordSpace; see [Mereology.md](Mereology.md).

### Slot selectors (`what` / `where` / `when`) — RETIRED

Retired 2026-05-04: subspace partitioning is the dispatcher's
responsibility (the modality tensors carry the slot structure
intrinsically), not a grammar rule.

### Absorb — base-class marker

`absorb(\ell, r) = \ell`.  Binary left-pass; the right operand is
sugar that the rule predictor opted to discard.  No longer a
dedicated layer class -- the marker lives on
``GrammarLayer.absorb(left, right) -> left`` so any binary subclass
can flag its right operand as syntactic sugar without adding a
distinct dispatch entry.

### Query — Mereological

`query(\ell, r) = \ell` (accumulator preservation).  The query
marker is pushed onto WordSubSpace at the chart's accumulation point
when a norm-drop fires (see [Composition](#composition) below).

### Swap (Sinkhorn soft permutation) — Sym

A learned $3 \times 3$ logit matrix $M$ is normalized via $k=5$
Sinkhorn iterations to produce a doubly-stochastic soft permutation
$P$.  Given operands $\ell, r$ and a learned broadcast marker $m$,
the output is the first row of $P \cdot [\ell, r, m]^\top$.  Lossy.

### Lift / lower — SigmaPi (mode-dispatch ops)

`lift` and `lower` are unified mode-dispatchers.  Default modes are
`lift` $\to$ `'OR'` (synthesis, $\vee$) and `lower` $\to$ `'AND'`
(analysis, $\wedge$).  The forward bodies follow the table above
under [conjunction / disjunction](#conjunction--disjunction--sym);
region-form inputs (tuples $(\ell, u)$) emit the obvious bound
intervals.

Cross-mode delegation: `lift(mode='AND')` forwards to
`lower(mode='AND')`; `lower(mode='OR')` forwards to
`lift(mode='OR')`.  `mode='NOT'` is `Ops.negation(X1)` on either
dispatcher.

`Ops.lowerReverseAll(Y, W, monotonic)` and `Ops.liftReverseAll(Y, W,
monotonic)` return the multi-operand pair `(recovered_left,
recovered_right)`.  The single-operand analytic forms
$\mathrm{Ops.liftReverse}(\mathrm{result}, \mathrm{right}) = \mathrm{result} / (\mathrm{right} + \varepsilon)$
and $\mathrm{Ops.lowerReverse}(\mathrm{result}, \mathrm{right}) = 2 \cdot \mathrm{result} - \mathrm{right}$
are preserved as legacy callers' inverse of the old
$\mathrm{Ops.lift} = \odot$ and $\mathrm{Ops.lower} = \mathrm{mean}$
bodies.

The cfg-driven grammar dispatch invokes `Ops.lift` / `Ops.lower`
directly.  The learnable equivalents live on
`SymbolicSpace.sigma.forward` / `ConceptualSpace.pi.forward`, which
own their own log-domain (Pi) and atanh-domain (Sigma) matmul
bodies.  The bridge is the `binary=True` flag on
`PiLayer.forward` / `SigmaLayer.forward`: it pre-applies
`Ops.top2_select_ste` to hard-select the top-2 input operands
before the layer body runs.

### Chunking — Percepts

Chunking is not a grammar rule.  It is a perceptual pre-pass
implemented by `ChunkLayer` / `PerceptualSpace.chunk_static`, before
the grammar sees symbol slots.  No `chunk` entry exists in
`GRAMMAR_LAYER_CLASSES`.

### Mereological suite — see [Mereology.md](Mereology.md)

The mereology suite (`part`, `whole`, `equal`, `overlap`, `underlap`,
`boundary`, `copart`) lives in `Ops.*`; the canonical specification
of its semantics is [Mereology.md](Mereology.md).  Each member has a
vector form (default) and a scalar form (`scalar=True`).  `part` and
`equals` are surfaced through grammar dispatch as `PartLayer` /
`EqualsLayer`; the rest are utility-level.

Empty-set conventions (parthood scalar form):
$\mathrm{part}(\emptyset, y) = 1$,
$\mathrm{part}(x, \emptyset) = 0$,
$\mathrm{part}(\emptyset, \emptyset) = 1$.

---

## GrammarLayer Implementations

`GrammarLayer` (in `bin/Layers.py`) is the base class for layers that
implement grammar rule operators.  Each subclass declares
`rule_name`, `arity`, `invertible`, and `lossy` as class attributes,
then overrides `forward()` (and `reverse()` if `invertible`).
Binary subclasses additionally override `compose(left, right)` and
`generate(parent)` for the chart's binary tensor interface; the
defaults route arity-1 layers through `forward` / `reverse` and raise
on arity-2 unless overridden.

The hardcoded module-level `GRAMMAR_LAYER_CLASSES` dict at the bottom
of the section maps `rule_name` to subclass.  Per-space dispatchers
look up rules through this table when the grammar references a rule
whose host parametrized layer isn't already owned by the space.

### Base contract

```python
class GrammarLayer(Layer):
    rule_name  = ""
    arity      = 1
    invertible = False
    lossy      = False
    _chart_authority = None  # set via set_chart_authority(syntactic_layer)

    def gated_run(self, x, fn, *fn_args, **fn_kwargs):
        auth = GrammarLayer._chart_authority
        if auth is None or not self.rule_name:
            return fn(x, *fn_args, **fn_kwargs)
        try:
            p = float(auth.should_run_rule(self.rule_name))
        except Exception:
            return fn(x, *fn_args, **fn_kwargs)
        if p >= 1.0 - 1e-9:
            return fn(x, *fn_args, **fn_kwargs)
        if p <= 1e-9:
            return x
        return p * fn(x, *fn_args, **fn_kwargs) + (1.0 - p) * x

    def compose(self, *operands):
        if self.arity == 1:
            return self.forward(operands[0])
        raise NotImplementedError

    def generate(self, parent):
        if self.arity == 1:
            return self.reverse(parent) if self.invertible else parent
        raise NotImplementedError

    def decompose(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
```

### `NotLayer` — `S = not(S)`

Tier `S`.  Bivector pole swap on ``[..., :2]`` of the muxed event;
``[..., 2:]`` (nWhere / nWhen) passes through.  Self-inverse.

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

### `NonLayer` — `S = non(S)`

Tier `S`.  Per-pole bivector complement on ``[..., :2]``:
``[1 - pos, 1 - neg]``.  Self-inverse on each pole; ``[..., 2:]``
tail passes through.

```python
class NonLayer(GrammarLayer):
    rule_name  = "non"
    arity      = 1
    invertible = True

    def forward(self, x):
        bivector = 1.0 - x[..., :2]
        rest     = x[..., 2:]
        return torch.cat([bivector, rest], dim=-1) if rest.shape[-1] else bivector

    def reverse(self, y):
        return self.forward(y)
```

### `IntersectionLayer` — L-tier lattice min on bivector activation

Tier `L`.  Lattice min on a bivector activation ``[..., 2]``.
``reads_activation = True`` so the dispatcher feeds
``subspace.materialize(mode='activation')`` rather than the muxed
event.  ``monotonic`` toggles between RadMin (zero passthrough) and
strict min.

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

### `UnionLayer` — L-tier lattice max on bivector activation

Tier `L`.  Counterpart to ``IntersectionLayer``: lattice max on a
bivector activation, RadMax / strict-max toggle.

```python
class UnionLayer(GrammarLayer):
    rule_name        = "union"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'L'
    reads_activation = True

    def __init__(self, monotonic=False):
        super().__init__(0, 0)
        self.monotonic = bool(monotonic)

    def forward(self, left, right):
        return Ops.union(left, right, monotonic=self.monotonic)

    def reverse(self, parent):
        return parent, parent
```

### `ConjunctionLayer` / `DisjunctionLayer` — S-tier monotonic on scalar activation

Tier `S`.  Operate on the **post-codebook scalar activation**:
``[B, V]`` non-negative strength per prototype, returned by
``effective_activation()`` (bivector reduced via ``max`` and gated
by modal presence).  Strictly monotonic by construction -- there is
no negative pole on the scalar, so RadMin / RadMax would be wrong.

```python
class ConjunctionLayer(GrammarLayer):
    rule_name        = "conjunction"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'S'
    reads_activation = True

    def forward(self, left, right):
        return Ops.intersection(left, right, monotonic=True)

    def reverse(self, parent):
        return parent, parent


class DisjunctionLayer(GrammarLayer):
    rule_name        = "disjunction"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'S'
    reads_activation = True

    def forward(self, left, right):
        return Ops.union(left, right, monotonic=True)

    def reverse(self, parent):
        return parent, parent
```

### `LiftLayer` / `LowerLayer` — S-tier gated SVD slice on muxed event

Tier `S`.  ``LiftLayer`` runs ``sigma.compose`` (an OR fold) gated
by a per-rule ``raw_gate`` parameter; ``LowerLayer`` runs
``pi.compose`` (an AND fold).  Both share one matrix per type
(``space.sigma_S`` / ``space.pi_S``) lazy-constructed when the
grammar references their rule, with one gate per rule on the SVD
diagonal.  The reverse uses the analytical LDU inverse of the
gated matrix (``sigma.generate`` / ``pi.generate``), which is
exact when the gate is stable.

Both read the muxed event (``reads_activation = False``) since the
gated SVD slice operates on the full ``[B, V, D]`` symbolic event
tensor, not just the activation poles.

### `EqualsLayer` / `PartLayer` — mereology

`equals` weights the right operand by mutual parthood; `part` weights
by directional parthood.  Both broadcast a scalar score back to the
right operand and are lossy.

```python
class EqualsLayer(GrammarLayer):
    rule_name  = "equals"
    arity      = 2
    invertible = False
    lossy      = True

    def forward(self, left, right):
        score = Ops._equal_kernel(left, right, scalar=True)
        while score.ndim < right.ndim:
            score = score.unsqueeze(-1)
        return score * right

    def reverse(self, parent):
        return parent, parent

    def compose(self, left, right):
        return self.forward(left, right)

    def generate(self, parent):
        return self.reverse(parent)


class PartLayer(GrammarLayer):
    rule_name  = "part"
    arity      = 2
    invertible = False
    lossy      = True

    def forward(self, left, right):
        score = Ops._part_kernel(left, right, scalar=True)
        while score.ndim < right.ndim:
            score = score.unsqueeze(-1)
        return score * right

    def reverse(self, parent):
        return parent, parent

    def compose(self, left, right):
        return self.forward(left, right)

    def generate(self, parent):
        return self.reverse(parent)
```

### `TrueLayer` / `FalseLayer` — pole projection

```python
class TrueLayer(GrammarLayer):
    rule_name  = "true"
    arity      = 1
    invertible = False
    lossy      = True

    def forward(self, x):
        return torch.relu(torch.clamp(x, -1.0, 1.0))

    def reverse(self, y):
        return y


class FalseLayer(GrammarLayer):
    rule_name  = "false"
    arity      = 1
    invertible = False
    lossy      = True

    def forward(self, x):
        return torch.relu(-torch.clamp(x, -1.0, 1.0))

    def reverse(self, y):
        return y
```

### `SwapLayer` — `S = swap(S, S)`

Sinkhorn-normalized soft permutation.  The `swap_logits` and
`swap_marker` parameters live on this layer.

```python
class SwapLayer(GrammarLayer):
    rule_name  = "swap"
    arity      = 2
    invertible = False
    lossy      = True

    def __init__(self, swap_size=1, sinkhorn_iters=5):
        super().__init__(0, 0)
        self.swap_size = max(int(swap_size), 1)
        self._sinkhorn_iters = int(sinkhorn_iters)
        self.swap_marker = nn.Parameter(torch.randn(self.swap_size) * 0.01)
        self.swap_logits = nn.Parameter(torch.zeros(3, 3))

    def _soft_perm(self):
        M = self.swap_logits
        for _ in range(self._sinkhorn_iters):
            M = M - M.logsumexp(dim=-1, keepdim=True)
            M = M - M.logsumexp(dim=-2, keepdim=True)
        return M.exp()

    def forward(self, left, right):
        P = self._soft_perm()
        marker = self.swap_marker.to(left.device)
        if left.ndim == 3:
            D = left.shape[-1]
            m = marker[:D].unsqueeze(0).unsqueeze(0).expand_as(left)
        elif left.ndim == 2:
            m = marker[:left.shape[-1]].unsqueeze(0).expand_as(left)
        else:
            m = marker
        if right is None:
            right = left
        stack = torch.stack([left, right, m], dim=0)
        out = torch.einsum('ij,j...->i...', P, stack)
        return out[0]

    def reverse(self, parent):
        return parent, parent

    def compose(self, left, right):
        return self.forward(left, right)

    def generate(self, parent):
        return self.reverse(parent)
```

### `QueryLayer` — `S = query(S, S)`

Returns the left operand unchanged (the chart's query semantic
pushes a marker word elsewhere; the rule's tensor contribution is
the accumulator passthrough).

```python
class QueryLayer(GrammarLayer):
    rule_name  = "query"
    arity      = 2
    invertible = False

    def forward(self, left, right):
        return left

    def reverse(self, parent):
        return parent, parent

    def compose(self, left, right):
        return self.forward(left, right)

    def generate(self, parent):
        return self.reverse(parent)
```

### Retired: `WhatLayer` / `WhereLayer` / `WhenLayer` / `AbsorbLayer`

Removed 2026-05-04.  Slot partitioning is now the dispatcher's
responsibility (the modality tensors carry the slot structure
intrinsically), so ``what`` / ``where`` / ``when`` are not grammar
ops.  ``absorb`` became a base-class marker:
``GrammarLayer.absorb(left, right) -> left`` lets any binary subclass
flag its right operand as syntactic sugar without a dedicated
dispatch entry.

### `PiLayer` — parametrized AND fold

`PiLayer` (and `SigmaLayer` below) inherit from `ButterflyLayer`, an
intermediate base that adds the optional butterfly access pattern
on top of `GrammarLayer`.  The unary `forward(x)` is the log-domain
multiplicative AND fold over the feature axis; the binary
`compose(left, right)` runs the same fold but sums log-mult contri-
butions across operands instead of within one.

```python
class PiLayer(ButterflyLayer):
    rule_name  = "intersection"
    arity      = 2

    def forward(self, x, binary=False):
        if self.butterfly:
            B, N, D = x.shape
            packed = self._butterfly_pack(x)
            packed_out = self._pi_inner_forward(packed, binary=binary)
            x_out = self._butterfly_unpack(packed_out, B, N, D)
            return self._butterfly_merge(x_out)
        return self._pi_inner_forward(x, binary=binary)

    def reverse(self, y):
        if self.butterfly:
            x_out = self._butterfly_unmerge(y)
            B, N, D = x_out.shape
            packed = self._butterfly_pack(x_out)
            packed_in = self._pi_inner_reverse(packed)
            return self._butterfly_unpack(packed_in, B, N, D)
        return self._pi_inner_reverse(y)

    def compose(self, left, right):
        if self.butterfly:
            raise NotImplementedError("PiLayer.compose not supported in butterfly mode")
        if self.layer.ergodic:
            self.resample_noise()
        W = self.layer.compute_W_current()
        left = left.to(W.device); right = right.to(W.device)
        if self.nonlinear:
            l_l = torch.log(self._to_mult(left))
            l_r = torch.log(self._to_mult(right))
        else:
            l_l = torch.log(left); l_r = torch.log(right)
        wl = (l_l + l_r) @ W + self.layer._effective_bias()
        return torch.tanh(wl / 2) if self.nonlinear else torch.exp(wl)

    def generate(self, parent):
        if self.butterfly:
            raise NotImplementedError("PiLayer.generate not supported in butterfly mode")
        W_inv = self.layer.compute_Winverse_current()
        parent = parent.to(W_inv.device)
        log_mult_y = torch.log(self._to_mult(parent)) if self.nonlinear else torch.log(parent)
        s = (log_mult_y - self.layer._effective_bias()) @ W_inv
        half = s * 0.5
        op = self._from_mult(torch.exp(half)) if self.nonlinear else torch.exp(half)
        if self.layer.ergodic:
            self.resample_noise()
        return op, op
```

`_pi_inner_forward` / `_pi_inner_reverse` are the shared kernels:
`(1+x)/(1-x)` to mult-domain $\to$ log $\to$ linear $\to$ tanh-half $\to$ back to
$[-1, 1]$.  `binary=True` pre-applies `Ops.top2_select_ste` to
hard-select the top-2 input operands.

### `SigmaLayer` — parametrized OR fold

```python
class SigmaLayer(ButterflyLayer):
    rule_name  = "union"
    arity      = 2

    def forward(self, x, binary=False):
        if self.butterfly:
            B, N, D = x.shape
            packed = self._butterfly_pack(x)
            packed_out = self._sigma_inner_forward(packed, binary=binary)
            x_out = self._butterfly_unpack(packed_out, B, N, D)
            return self._butterfly_merge(x_out)
        if binary:
            x = Ops.top2_select_ste(x)
        if self.nonlinear:
            x = torch.atanh(x.clamp(-1 + epsilon, 1 - epsilon))
        y = self.layer.forward(x)
        if self.nonlinear:
            y = torch.tanh(y)
        self.activation = y.detach()
        return y

    def reverse(self, y):
        if self.butterfly:
            x_out = self._butterfly_unmerge(y)
            B, N, D = x_out.shape
            packed = self._butterfly_pack(x_out)
            packed_in = self._sigma_inner_reverse(packed)
            return self._butterfly_unpack(packed_in, B, N, D)
        if self.nonlinear:
            y = torch.atanh(y.clamp(-1 + epsilon, 1 - epsilon))
        x = self.layer.reverse(y)
        if self.nonlinear:
            x = torch.tanh(x)
        self.activation = x.detach()
        return x

    def compose(self, left, right):
        if self.butterfly:
            raise NotImplementedError("SigmaLayer.compose not supported in butterfly mode")
        if self.nonlinear:
            a_l = torch.atanh(left.clamp(-1 + epsilon, 1 - epsilon))
            a_r = torch.atanh(right.clamp(-1 + epsilon, 1 - epsilon))
        else:
            a_l, a_r = left, right
        out = self.layer.forward(a_l + a_r)
        if self.nonlinear:
            out = torch.tanh(out)
        self.activation = out.detach()
        return out

    def generate(self, parent):
        if self.butterfly:
            raise NotImplementedError("SigmaLayer.generate not supported in butterfly mode")
        a_y = (torch.atanh(parent.clamp(-1 + epsilon, 1 - epsilon))
               if self.nonlinear else parent)
        a_sum = self.layer.reverse(a_y)
        half = a_sum * 0.5
        op = torch.tanh(half) if self.nonlinear else half
        return op, op
```

The `OR`-fold is the additive (logit-domain) sum: atanh the operands,
sum across operands, then apply the same linear + tanh.  `compose` /
`generate` close the loop with a balanced inverse so the chart can
push parent gradients back into child cells.

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
    # L-tier (logical -- lattice on bivector activation)
    'intersection': IntersectionLayer,
    'union':        UnionLayer,
}
```

### Inverse contract by operator

| Op            | Tier | Reverse                                        |
|---------------|------|-------------------------------------------------|
| `not`         | S    | exact (self-inverse pole flip)                 |
| `non`         | S    | exact (self-inverse: ``non(non(x)) = x``)      |
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

Reverse / inverse Ops (`Ops.negationReverse`, `Ops.conjunctionReverse`,
`Ops.disjunctionReverse`, `Ops.liftReverse`, `Ops.liftReverseAll`,
`Ops.lowerReverse`, `Ops.lowerReverseAll`) are not in the dispatch
registry — they are invoked directly by the Step-6 grammar reverse
convention or by `Basis.*` methods.  The
`_binary_op_inverse_impl(result, W, op, monotonic)` private helper
backs both `conjunctionReverse` and `disjunctionReverse` via codebook
search.

---

## Rule prediction

`SyntacticLayer.predict_rules(x)` — also exposed on `Chart` after the
2026-05-01 refactor — predicts a rule distribution at each derivation
depth.  The architecture is weight-tied (recursive) across depths
with a learned depth embedding.

Input projection:

$$
h^{(0)} = \sigma(W_{\text{in}}\, x + b_{\text{in}}),
\qquad h^{(0)} \in \mathbb{R}^{B \times d_{\text{hidden}}}
$$

For each depth $d = 0, 1, \ldots, D_{\max} - 1$ (shared weights
$W_{\text{deriv}}$, $W_{\text{head}}$):

$$
\begin{aligned}
h^{(d+1)} &= \sigma\!\bigl(W_{\text{deriv}}\,(h^{(d)} + e_d) + b_{\text{deriv}}\bigr) \\
z^{(d)}   &= W_{\text{head}}\,h^{(d+1)} + b_{\text{head}} \in \mathbb{R}^{B \times K} \\
z^{(d)}_t &\leftarrow \mathrm{stop\_grad}\bigl(z^{(d)}_t\bigr) + (1 - \iota)\cdot \tau_{\text{trans}}
           \quad \text{(transition-bias on rule } t\text{)} \\
p^{(d)}   &= \begin{cases}
              \mathrm{gumbel\_softmax}(z^{(d)}; \tau) & \text{training} \\
              \mathrm{softmax}(z^{(d)})               & \text{eval}
             \end{cases}
\end{aligned}
$$

where $e_d$ is the depth embedding,
$\iota = \mathrm{grammar.interpretation}$, $t$ is the index of the
transition rule (if present), and $\tau$ is the Gumbel-softmax
temperature (annealed from 1.0 toward 0.1 via `set_tau`).

Output: `{rule_logits, rule_probs, predicted_rules}`.  Training uses
gumbel-softmax (soft one-hot, gradients flow through all candidate
rules); eval uses softmax + argmax (discrete selection).

---

## Composition

Chart composition lives on the `Chart` class (`bin/Language.py`).
`Chart.compose(data, word_space, subspace=None)` runs the inside pass
over `data` and populates per-tier rule selections on
`word_space.current_rules`.  `Chart.generate(target, word_space,
subspace=None)` runs the outside pass + Viterbi backtrace and
populates `word_space.generate_rules`.

The chart router is selected via `<WordSpace><routerKind>` (XML);
either `'chart'` (the default soft-superposition CKY) or `'signal'`
(`SignalRouter`, a tensorial alternative).  Both surfaces produce the
same per-tier rule lists.

### Phase 1: deterministic `not` (legacy 2D path)

Before soft superposition, the legacy 2D-activation path applies a
single deterministic `not` to the top-of-stack when that vector has a
negative mean, and pushes a `not` word for that batch row.  This
strips any leading negation from the accumulator before it mixes
with siblings.  The 3D chart path skips this step — sign flips go
through `NotLayer.forward` like every other rule.

### Phase 2: chart inside pass

For each cell $(i, j)$ in the chart and each grammar rule with arity
matching the cell's shape, the chart computes:

$$
\mathrm{score}_{ij}^{\rho} = \mathrm{compat}(\text{lhs}, \text{rhs}_{ij}) \cdot
                              \mathrm{pair\_score}(i, j) \cdot
                              \mathrm{rule\_logit}(\rho)
$$

then takes the soft mixture (training) or argmax (eval) over all
firing rules at that cell.  Per-cell rule applications dispatch
through `wordSpace.host_layer(tier, rule_name)`, which returns the
GrammarLayer instance owned by the host space.  The chart then calls
`layer.compose(left, right)` (binary) or `layer.forward(x)` (unary).

The compatibility mask `compat[B, P, R]` zeroes out `(pair, rule)`
combinations whose typed LHS / RHS categories don't match the pair's
slot categories.  Function-call rules (legacy, `rhs_symbols = None`)
are always compatible.  Category `0` (`'?'`) is a wildcard, so
unseeded leaves can match any typed rule during warmup.

#### Query / norm-drop detection

Detect symbolic contradiction at the accumulation point by comparing
the pre- and post-composition Frobenius norms per batch row.  Define
$\nu_\ell = \|\ell\|_F$ and $\nu_c = \|c\|_F$.  When $\nu_\ell >
10^{-6}$ and $\nu_c < \kappa\,\nu_\ell$ with $\kappa = 0.1$
(`_QUERY_NORM_DROP_RATIO`), the new accumulator preserves the prior
state and a `query` word is pushed onto WordSubSpace.  Otherwise the
chart accepts the candidate.

The argmax rule is recorded as a word at the leaf position;
`last_rule_per_batch[b]` is updated only when the query mask did
*not* fire (so `query` preserves the prior rule context).

### Tensor word buffer

Each `SubSpace` carries two registered buffers alongside the legacy
`self.word: list[tuple]`:

```python
self.word_records  # [B*K, max_depth, ENTRY_WIDTH] long, ENTRY_WIDTH=7
self.word_count    # [B*K] long, current depth per cell
```

`add_word(...)` is overloaded.  The **scalar form**
(`add_word(int, int, int, ...)`) appends one validated 7-tuple to
`self.word`; used by direct callers (tests, the legacy compose path).
The **vector form** (`add_word(LongTensor, ...)`) scatters into
`word_records` at each cell's current `word_count`, then increments
`word_count` (no host sync).

The chart materializes its tensor buffer before `_compose_chart_cky`
returns, by calling `subspace.flush_word_buffer()` inside the chart
path.  That keeps direct compose callers, the SVO walker, and
derivation-trace tests seeing `self.word` immediately populated in
cell-major order.  This is "Path B" hybrid tensor buffering with host
materialization — not yet the full tensor-only Path A.

### Chart-derived SVO

`Chart._extract_svo_from_trace` walks each batch row's trace after
chart compose finishes:

1. Find the last `S -> S VO` entry (outermost reduction).
2. Find the matching `VO -> V O` entry whose `merged_slot` equals the
   outer rule's right arg.
3. Pull `subject` from the outer's left slot, `verb` from the inner's
   left, `object` from the inner's right.

Each is a `[1, D]` slice of the original pre-compose `data`.  Rows
that don't contain the canonical pair of firings get a zero `[1, D]`
placeholder; if every row in a batch fails, `last_svo` stays `None`.

---

## POS side-channel

A part-of-speech distribution rides alongside the signal at every
chart cell.  The trailing axis of the side-channel is `|POS| =
|Grammar.categories|` — the union of every `lhs` / `rhs_symbol`
declared by the grammar (e.g. `S`, `VO`, `NP`, `VP`, `V`, `N`,
`ADJ`, `DET`).  Index 0 is `'?'`, the wildcard.

### Storage tiers

- **Per-atom POS tag** (durable, seed source) lives on the
  SymbolicSpace codebook as `Codebook.category_ids: Long[V]` (one
  long index per atom row, default `0` = `'?'`).  Allocated when
  the codebook is built with `category=True`.  Mirrors the existing
  `polarity_ids` pattern.  Use `set_category(idx, cat_id)` /
  `get_category(idx)` to read / write per atom.
- **Per-call chart POS** is `Chart._chart_pos: [B, N+1, N+1, |POS|]`,
  the softmax of `Chart._chart_score` along the trailing axis.  Each
  cell row is a probability simplex.  Cleared at the start of every
  inside pass.
- **Durable per-merge POS** lives on `SubSpace.pos_records:
  Long[B*K, max_depth]`, parallel to `word_records` / `word_count`.
  Each entry is the merged-cell POS at the depth recorded by the
  matching `word_records` row.

### Lexical bootstrap

`Chart._apply_codebook_pos_seed` overrides the learned
`_lex_cat_scorer(data)` log-distribution with a one-hot at the
codebook-resolved POS for any input position whose nearest atom
carries a non-wildcard `category_ids` entry and whose dot-product
similarity exceeds `0.1`.  Untagged or low-confidence positions
retain `softmax(_lex_cat_scorer(data))`.  Gradients still flow
through `_lex_cat_scorer` on those positions.

### Mechanism 1 — RHS POS compatibility mask

In `_chart_inside`, before scoring each binary merge candidate, the
chart gathers the operands' POS distributions
(`pos_left[..., rule.rhs_left[r]]`, `pos_right[..., rule.rhs_right[r]]`)
and adds `log(p_left * p_right)` to the rule's candidate score.
Wildcard rules (`rhs_*[r] == 0`) get an unconditional `1.0`
multiplier.  For typed rules like `NP = Intersection(ADJ, N)`, this
makes the rule fire only when the left operand looks like `ADJ` and
the right looks like `N`; the score for `[N, ADJ]` is exponentially
suppressed.

### Mechanism 3 — Rule-prediction conditioning

After `_viterbi_extract` runs, `Chart._populate_category_stack`
walks the trace and pushes each merge's LHS category embedding
(from `WordSpace.category_codebook.W`) onto
`WordSpace.category_stack`.  Subsequent calls to
`WordSpace.predict_rule(b)` then read a parse history of POS
embeddings via the existing
`category_stack.flatten(b)` $\to$
`Linear(max_depth * pos_dim $\to$ n_rules)` MLP that was already
allocated for this purpose.

### Why this trains the network to assign POS to every word

Two reinforcing gradient paths:

1. **Anchored** — codebook atoms with a known category override
   `_lex_cat_scorer` deterministically.  Their POS distributions
   propagate up the chart and gate Mechanism 1; reconstruction loss
   pulls the model's downstream rule choices toward those atoms'
   stored categories.
2. **Emergent** — untagged atoms fall back to
   `softmax(_lex_cat_scorer(data))`.  A wrong POS at a leaf
   produces a wrong rule firing (RHS mask fails, rule scores
   drop), bad reconstruction, and a strong gradient back through
   `_lex_cat_scorer.W` to fix the leaf.

The two combine: tagged atoms anchor a stable POS reference; the
scorer learns to extend that reference to untagged atoms whose
context makes the correct POS unambiguous.  Coverage matters —
tagging the closed-class words (DET, AUX, PREP, CONJ) plus the
most frequent open-class atoms gives the system a working anchor
set.

### `<writeSyntax>true</writeSyntax>` — syntax-tree dump

When `<architecture><writeSyntax>true</writeSyntax></architecture>`
is set in the model XML, `BasicModel.forward()` calls
`BasicModel.write_syntax_tree(path)` at the end of every forward
pass.  The output path defaults to `output/syntax.xml` and is
overridable via
`<architecture><syntaxOutPath>...</syntaxOutPath></architecture>`.

The dumper walks the chart's `_derivation_trace` and the symbolic
codebook to emit one `<forward>` XML fragment per call, with one
`<batch>` per row.  Category names (`cat="..."`), rule canonicals
(`rule="..."`), and POS tags (`pos="..."`) come straight from the
live grammar — whatever `Grammar.categories` and `RuleDef.canonical`
hold at parse time.  For example, given the XOR grammar
([data/XOR_grammar.xml](../data/XOR_grammar.xml)) which declares
`S = not(S) | conjunction(S, S) | disjunction(S, S)` over a single
non-terminal `S`, a parse of the input `1 1` (true $\wedge$ true) under the
CKY chart would emit something like:

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

A grammar that adds typed categories (e.g. `NP = intersection(ADJ, N)`)
would produce richer trees with those category names appearing on
the `<node>` and `<leaf>` elements; the dumper has no built-in
category vocabulary of its own.

Each `<node>` carries `cat="<lhs_category>"`,
`rule="<rule_canonical>"`, and `i`/`j` for the span; each `<leaf>`
carries `token`, `pos` (resolved through
`SymbolicSpace.subspace.what.category_ids`, falling back to `'?'`
when the atom carries no tag), and the input position `i`.  The
`<rules>` element lists the global rule-id sequence for compact
diff-based regression.

The file is truncated on the first call within a process and
appended on subsequent calls so a training run accumulates one
`<forward>` fragment per forward pass.  Wrap the file in a single
`<syntaxLog>` root element (or use a streaming XML parser) if a
single valid XML document is required.  `<writeSyntax>` defaults
to `false`; the tree-builder is fully bypassed in that case (no
per-forward overhead).

Note: when the parser's `<routerKind>signal</routerKind>` is
selected, the SignalRouter bypasses `Chart._chart_inside` and
no derivation trace is recorded; the dumper emits
`<noTrace/>` for those rows.  Set `<routerKind>chart</routerKind>`
(the default) to use the CKY inside pass that populates the
trace.

---

## Per-space dispatch (`SyntacticLayer`)

Each Space (SymbolicSpace, ConceptualSpace, PerceptualSpace) carries a
`SyntacticLayer` instance whose `forward()` and `reverse()` fire
one fold step per call, per the chart's per-tier rule selection.
Construction lives in `build_space_syntactic_layer`:

- `host_layers`: dict mapping `rule_name` to a `GrammarLayer` instance
  (parametrized folds like `space.pi` for `'intersection'`,
  `space.sigma` for `'union'`, plus stateless rules built lazily from
  `GRAMMAR_LAYER_CLASSES`).
- Each `host_layer` is registered with the WordSpace via
  `register_host_layer(tier, rule_name, layer)` so the chart can
  consult `host_layer(tier, rule_name)` during dispatch.

`SyntacticLayer.forward(subspace)` reads
`word_space.current_rules[tier]`, advances a per-tier cursor, and
dispatches to the chosen layer's `forward()` (only for arity-1 rules
— binary rules fire inside the chart's `compose` via
`host_layer.compose(left, right)`).  `reverse()` mirrors via
`word_space.generate_rules` and `layer.reverse()`.  The cursor resets
at the start of each new `word_space.compose()` /
`word_space.generate()` call via the WordSpace generation counters.

When the chart router is `'signal'`, the per-rule unary fold is
skipped — the SignalRouter has already executed the derivation
tensorially and written the `[B, 1, D]` root state back into the
subspace.

### `WordSpace.forwardSymbols` / `reverseSymbols`

These methods used to invoke a private composition pipeline.  After
the 2026-05-01 refactor they are thin: `forwardSymbols(data,
subspace)` demuxes the muxed symbol tensor into the subspace's
modality slots when shapes match (the side-effect downstream slot
selectors depend on); the actual symbolic composition runs on the
chart via the `ChartCompose` pipeline stage.  `reverseSymbols` is a
pass-through; chart-driven generation handles the symbol-side reverse
via `ChartGenerate` + per-space `SyntacticLayer.reverse` dispatch.

---

## Downward head emission (`S -> C`)

Selected by `<WordSpace><downwardGeneration>true</downwardGeneration>`.
After the upward parse reduces `N` leaves to a single root vector,
`BasicModel.forward` takes that root (`sym_vectors[:, 0, :]`) and
calls `WordSpace.reconstruct(state, inputSpace)`, which runs the
downward `S -> C` rule as a one-shot projection onto a codebook.

### Projection-coefficient scoring

$$
W_{\text{norm}} = W / \lVert W\rVert_{2,\text{row-wise}},
\qquad
\text{scores} = \mathrm{clamp}_{\ge 0}(\text{state} \cdot W_{\text{norm}}^\top)
$$

The per-row argmax is `best_idx`.  Because `state` is *not*
L2-normalized (only the codebook rows are), the score is "how much of
atom $k$ lives in the state" rather than a pure angle.  The resulting
`contained = scores[:, k] * W_{\text{norm}}[k]` is the slice of the
atom actually contained in the state; `residual = state - contained`
is the leftover meaning a future expansion step (NP / VP templates)
could consume.

### `WordSpace.reconstruct(state, codebook_space)`

Any space whose `.subspace.basis` exposes a `getW() -> [V, D]` works.
`InputSpace` (word embedding) is the intended source — projecting the
deep state onto word vectors picks the vocabulary entry that best
matches the sentence's composed meaning, and `.wv.index_to_key`
decodes the head index back to a token.  `SymbolicSpace` (internal
atom codebook) is also valid once a codebook is populated there.
When the codebook is unwired (passthrough `Codebook` with `getW() ==
None`), `reconstruct` returns a trivial `heads=[0]*B` so the loss
path never crashes.

Returns `{heads, contained, residual, state}`.  `BasicModel.forward`
stashes `heads` as `self._predicted_head` so a training loss can
compare against a supervised head token, and so `bin/interact_head.py`
can print the decoded word after each forward pass.

---

## Word encoding

Each word is a 7-tuple stored in `subspace.word` (the legacy list) and
mirrored in `subspace.word_records` (the tensor buffer).  Per-cell:
`(batch, vector, rule, leaf1, leaf2, leaf3, order)` where the
`leaf*` slots carry codebook indices for terminal words and child
slot indices for internal nodes.  Terminal words carry `order = -1`
and a codebook index in `leaf1`; the leaf ledger used by chart
generation is built from these entries.

The list is a derivation tree in pre-order: the first entry is the
root, and binary rules expand left-first.

---

## Decomposition

`Chart.generate(target, word_space, subspace)` reconstructs the
pre-compose tensor from the symbolic word record by walking the
chart's outside pass + Viterbi backtrace.  At each step it dispatches
through `host_layer.generate(parent)` for binary rules or
`host_layer.reverse(y)` for unary invertible rules.  Lossy rules
(`equals`, `part`, `true`, `false`, `query`, `swap`, `conjunction`,
`disjunction`, `intersection`, `union`) emit their pseudo-inverses
(typically the parent passed back unchanged).

The codebook fast path: if a basis is available and shapes match, the
generator looks up each terminal word (those with `order = -1`) and
places `cb[leaf1]` at the recorded position, producing the exact
pre-compose tensor.

---

## How syntax is trained

The syntax code is end-to-end differentiable: `Chart.predict_rules`
produces soft `rule_probs`; the chart inside pass uses soft
superposition over compatible rules; the binary `compose` /
`generate` paths flow gradients through the parametrized fold weights
(PiLayer / SigmaLayer / SwapLayer logits).  The deterministic `not`
phase 1 step in the legacy 2D path is non-differentiable but only
fires when the top-of-stack mean is negative.

`BasicModel.runBatch()` computes three losses: output loss, optional
reconstruction loss, optional embedding loss.  There is no explicit
syntax loss, parse-tree loss, or rule-label supervision.  Syntax
fitness emerges indirectly through the reconstruction loss: rules
that consistently support successful reconstruction accumulate
weight, while weaker alternatives diminish.

Under `MentalModel.xml`, syntax is enabled, the chart and per-space
dispatchers are instantiated, and rules are predicted, fired, and
recorded.  `BasicModel.forward()` goes through `SigmaLayer` rather
than invoking the chart pipeline directly; syntax is active as
computation and analysis but not yet placed on a strong loss path in
the stock training loop.

---

## Two-Layer logic

The logic system operates at two levels.  The **subsymbolic** layer
treats objects as vector sets $(B, N, D)$ interpreted as RBF /
luminosity fields: union is $\max(\ell, r)$ (strongest affirmation);
intersection is $\min(\ell, r)$ (shared commitment); negation is
$-x$ (antipodal opposition on the hypersphere); non is $\alpha x$
with $\alpha \in [0, 1)$ (contraction toward zero); parthood is
fuzzy max-coverage in $[-1, 1]$ (signed containment).

A **symbolization** map projects vectors to scalar truth strength:

$$
s(X) = 2 \cdot \mathrm{mean}(\|x_i\|) - 1 \quad \in [-1, 1]
$$

Interpretation: $+1$ $\to$ strong presence, $0$ $\to$ neutral, $-1$ $\to$
absence.

The **symbolic** layer operates on scalars in $[-1, 1]$: neg is
$-a$, non is $\alpha a$, union is $\max(a, b)$, intersection is
$\min(a, b)$, part is $\mathrm{clamp}(b - a, -1, 1)$.

The key insight: subsymbolic = geometry; symbolic = order + polarity;
symbolization = norm projection.

---

## SymbolicSpace integration

### Forward path

1. Extract concept activation $[B, n_{\text{Concepts}}]$.
2. Map through `PiLayer(monotonic=True, invertible=True)` to
   $[B, n_{\text{Symbols}}]$.
3. Codebook quantization (when enabled) produces one-hot activation
   plus vectors.
4. The chart runs its inside pass over symbol vectors, dispatching
   per-cell rule applications through
   `wordSpace.host_layer('S', rule_name)` (which resolves to the
   SymbolicSpace's `SyntacticLayer`).

### Reverse path

1. Extract symbol activation $[B, n_{\text{Symbols}}]$.
2. Exact inverse of the PiLayer recovers $[B, n_{\text{Concepts}}]$.
3. The chart's outside pass reconstructs the pre-compose tensor from
   the word record via `host_layer('S', rule_name).generate(parent)`.

### Key properties

- Symbols are zero-dimensional — pure activation scalars, not
  vectors.
- The PiLayer allows $n_{\text{Concepts}} \neq n_{\text{Symbols}}$.
- The chart's per-cell soft superposition runs over all active
  S-tier rules; rule firing is gated by LHS / RHS POS compatibility
  via `chart_pos` (see [POS side-channel](#pos-side-channel)).

---

## AssociationLayer (EQUALS implementation)

The `AssociationLayer` is a cross-symbol associative memory used by
the EQUALS rule.  Two modes are available.  **`type="symmetric"`**
is Hopfield-like: it learns projection $A$, computes association
scores $A^\top A$, and softmax-retrieves the associated pattern;
associations are symmetric ($A \equiv B \Leftrightarrow B \equiv A$).
**`type="hopfield"`** is modern Hopfield: separate query / key
projections and softmax-gated retrieval.

Input and output are both $[B, N]$ activation vectors.  The layer is
learnable and its parameters are trained end-to-end via the
reconstruction loss.

---

## Future: RuleNode on the ReconstructionStack

A planned Phase 2 introduces a `RuleNode` structure with slots for
`(rule_id, method_name, forward_op, reverse_op, arity, args)` and
enriches `ReconstructionStack` to support variable-fidelity
reconstruction (pos-only; grammar + pos; grammar + pos + args).
This pairs generative rules with their compositional inverses so
downstream consumers can pull as much or as little reconstruction
context as they need from the same word record.

---

## Design history: learned SVO plan

This appendix preserves the original design narrative that justified
moving from a positional SVO tap to grammar-derived SVO roles.  The
plan was authored before the chart-compose implementation landed;
Phases A–D have since shipped.  Kept here for archaeology and as a
reference for any future extension of the same approach.

`TruthLayer.universality(subject, verb, object, lifting_layer,
symbolic_space)` implements the Golden Rule: it scores luminosity
change under S/O reversal, so $K(X, Y) + K(Y, X)$ illuminates more
than $K(X, Y)$ alone for kind actions.  The method takes
$(S, V, O)$ concept tensors as inputs; the question was where those
three tensors come from.

### Historical starting point (pre-chart)

SVO was produced by a positional tap inside
`SyntacticLayer._compose_vector`: the first three active leaf
positions of a batch row were labelled S/V/O when every row had at
least three active leaves.  For the canonical three-token transitive
corpus in `test/test_universality.py`, this produced the right roles.
It was wrong for realistic inputs: determiners shifted the roles,
adjectives inserted modifiers, word-order variation broke it
entirely, and embedded clauses were invisible.

The plumbing downstream of `last_svo` was already correct for a
learned tap to drop into:

- `Chart.last_svo: Optional[Tuple[Tensor, Tensor, Tensor]]`
- `SyntacticLayer.lifting_layer: LiftingLayer` (instantiated in
  `init_lifting`, called from `WordSpace._build_syntactic_layer`)
- `BasicModel.forward` reads both, calls
  `truth_layer.universality(...)`, and exposes the result via
  `self._universality_score`
- `truth_modulated_loss(universality_score=...)` folds the score into
  the training loss

Only the producer of `last_svo` needed to change.

### Phases summary

The five phases shipped:

- **Phase A — Grammar metadata.** `RuleDef` extended with
  `(lhs, rhs_symbols)` so each rule declares its typed nonterminal /
  terminal categories.  `Grammar.configure` iterates over every key
  in the XML `<grammar>` block; the bare-symbol-sequence form
  (`<S>S VO</S>`) was added as a typed compose alternative to the
  function-call form.
- **Phase B — Chart restructure.** The strict left-associative cascade
  was unable to produce `(S, (V, O))` from an `N V N` sentence (it
  forced step 1 to merge `leaf[0]` with `leaf[1]`).  The chart
  replaces it with per-step pair selection: a pair-scorer MLP picks
  *which two adjacent leaves* to merge, plus *which rule* to apply.
- **Phase C — Category tensor.** A `category: [B, N]` tensor rides
  alongside `data: [B, N, D]`.  At each merge, a compatibility mask
  zeros out `(pair, rule)` combinations whose typed RHS doesn't
  match the pair's slot categories.  Category `0` (`'?'`) is a
  wildcard so unseeded leaves match any typed rule during warmup.
- **Phase D — SVO extraction.** SVO falls out of the derivation
  trace.  After compose, find the outermost `S -> S VO` firing,
  find the matching `VO -> V O`, pull S from the outer's left arg
  and V / O from the inner's args.  Rows without the canonical pair
  get zero placeholders.
- **Phase E — Training dynamics.** Soft pair mixture, soft rule
  mixture, hard category argmax (with Gumbel-softmax to keep
  gradients flowing through "wrong" choices); curriculum from
  three-content-word SVO to longer / modified inputs.

Each phase is implemented in `bin/Language.py` (chart side) and
`bin/Layers.py` (per-rule layers).  The implementation tracker is
`doc/plans/2026-04-20-LearnedSVO-integration.md`; the chart-compose
section above describes the runtime behavior.

### Stable hooks already wired

- `Chart.last_svo: Optional[Tuple[Tensor, Tensor, Tensor]]` — set by
  `_extract_svo_from_trace` at the end of `_compose_chart_cky` /
  `_compose_chart_cky_viterbi`.
- `SyntacticLayer.lifting_layer: LiftingLayer` — created in
  `init_lifting`, called from `WordSpace._build_syntactic_layer`.
- `BasicModel.forward` reads both, invokes
  `truth_layer.universality(s, v, o, lifting_layer, symbolicSpace)`,
  and stores `self._universality_score`; `truth_modulated_loss`
  integrates the score into the training loss.
- `BasicModel._predicted_head: Optional[list[int]]` — set by the
  downward head emission described above.

These four interfaces stayed stable when learned SVO replaced
positional SVO.  The producer changed from the positional
`_compose_vector` tap to the chart trace walker.
