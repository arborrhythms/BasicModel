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
it is configured with **20 rules** total: 17 function-call upward rules
(IDs 0-16), 2 typed upward productions (IDs 17-18), and 1 downward
production (ID 19).  All rules are tier `'S'` -- `tier` is retained as a
routing artifact from the pre-merge era; new typed categories live on
the `lhs` field.

Function-call upward rules (IDs 0-16, `lhs = 'S'`):

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

Typed upward productions (IDs 17-18, bare-symbol form, `method = 'merge'`):

| Rule | Production | lhs | rhs_symbols |
|------|-----------|-----|-------------|
| 17 | $S \to S\ VO$ | `'S'`  | `('S', 'VO')` |
| 18 | $VO \to V\ O$ | `'VO'` | `('V', 'O')`  |

Downward production (ID 19, `method = 'emit_head'`):

| Rule | Production | lhs | rhs_symbols |
|------|-----------|-----|-------------|
| 19 | $S \to C$ | `'S'` | `('C',)` |

The dispatch table also knows `chunk` (an invertible max-union used by
the perceptual pre-pass), but `chunk` is not registered in the default
rule list.

### Accessors

- `Grammar.symbolic()` - indices of all S-tier rules
- `Grammar.symbolic_transition()` - `None` (no cross-tier transitions)
- `Grammar.binary_rules()` - indices of all arity-2 rules
- `Grammar.rule_by_id(i)` - canonical production string
- `Grammar.method_name(i)` - dispatch method name (e.g. `'swap'`)
- `Grammar.arity(i)` - 1 or 2
- `Grammar.tier(i)` - `'S'`

### Grammar Configuration

`Grammar.configure()` loads an XML `<grammar>` section (inside
`<WordSpace><language>`).  The block splits into `<upward>` (parsing /
chart compose), `<downward>` (generation from a deep symbolic state),
and a single `<start>` element naming the start symbol.

The `<start>S</start>` element is the canonical sentence boundary marker.
When `SyntacticLayer.compose` reduces a row's parse stack to a single node
of the start category, the layer emits a host-side soft-reset signal for
that row (`wordSpace._sentence_completed[b] = True`). The outer
doc-streaming loop drains this signal after each `runBatch` tick and
dispatches `wordSpace.soft_reset(batch=b)`. Soft reset clears every
per-sentence working buffer for row `b`: the parse stack rows
(`subspace.clear_rows(b*K, (b+1)*K)`), the category stack rows, the
reconstruction stack rows, `_last_svo[b]`, and re-arms `_stm_fired[b]`
so the next sentence fires its own discourse-prediction bias. Discourse
history (`InterSentenceLayer` ring buffer) and codebook EMA are *not*
cleared — they are document-scoped and form the inter-sentence prior
that bridges sentence N to sentence N+1 within the same document.
Hard reset (document boundary) wipes both per-sentence and discourse
state.

Without `<start>` declared, soft reset never fires; the cascade falls back
to hard reset only (per-document). This is the legacy behavior.
The loader accepts three RHS forms; the **explicit-op** form is the
canonical target per
[plans/2026-04-24-lift-lower-bivector-refactor.md](plans/2026-04-24-lift-lower-bivector-refactor.md)
Step 6 and
[specs/2026-04-24-lift-lower-bivector-design.md](specs/2026-04-24-lift-lower-bivector-design.md)
(its dispatch table maps each op name to a single
`Ops.lift` / `Ops.lower` / `Ops.equal` / ... call):

```xml
<grammar>
  <upward>
    <!-- (1) Explicit-op form (canonical per Step 6). The RHS *is* the   -->
    <!--     dispatched call; LHS may be a comma-separated tuple for     -->
    <!--     multi-output reverses.                                       -->
    <rule>S       = lift(NP, VP)</rule>
    <rule>NP      = intersection(AP, NP)</rule>
    <rule>S       = not(S)</rule>
    <rule>S,S     = intersection_inv(VO)</rule>     <!-- multi-output reverse -->

    <!-- (2) Function-call form (legacy S-tier ops). Tag = LHS, body =   -->
    <!--     op(args). Equivalent to (1) but uses the nonterminal as     -->
    <!--     the XML element name instead of a generic <rule> wrapper.    -->
    <S>not(S)</S>
    <S>swap(S, S)</S>

    <!-- (3) Bare-symbol form (transitional). RHS is a whitespace-       -->
    <!--     separated sequence of nonterminal / terminal categories;    -->
    <!--     dispatch method is the typed-compose `merge`.  Step 6 of   -->
    <!--     the lift/lower plan migrated `data/grammar.cfg` to the     -->
    <!--     explicit-op form (1); the previous bare-symbol cfg is      -->
    <!--     preserved at `data/grammar_legacy.cfg` for reference.       -->
    <S>S VP</S>
    <NP>AP NP</NP>
  </upward>
  <downward>
    <!-- One-shot head emission: project the deep state onto a          -->
    <!-- codebook and emit the best-match atom. Dispatch is              -->
    <!-- `emit_head`; no function-call form is accepted on this path.    -->
    <S>C</S>
  </downward>
</grammar>
```

The op names that may appear in the explicit-op RHS are the dispatch
keys for the unified
`Y = f(X1, X2=None, mode=..., kind=..., inverse=False)` signature
implemented in Step 1–2 of the refactor (see [Logic.md §8](Logic.md)
*Grammatical dispatch*):

| Op name in RHS                  | Resolves to                                                | Notes                                                              |
|---------------------------------|------------------------------------------------------------|--------------------------------------------------------------------|
| `lift(A, B)`                    | `Ops.lift(A, B, mode='OR')`                                | synthesis / disjunction (default `kind='strict'` → max)            |
| `lower(A, B)`                   | `Ops.lower(A, B, mode='AND')`                              | analysis / conjunction (default `kind='strict'` → min)             |
| `intersection(A, B)`            | `Ops.lower(A, B, mode='AND')`                              | meet — modification rules (`NP = intersection(AP, NP)`)            |
| `union(A, B)` / `disjunction`   | `Ops.lift(A, B, mode='OR')`                                | join — coordination                                                |
| `conjunction(A, B)`             | `Ops.lower(A, B, mode='AND')`                              | propositional $\land$ (alias of `intersection` at S-tier)          |
| `not(A)`                        | `Ops.lift(A, mode='NOT')`                                  | self-inverse pole flip / sign flip                                 |
| `equals(A, B)`                  | `Ops.equal(A, B)`                                          | mereological mutual parthood (copula identification)               |
| `part(A, B)`                    | `Ops.part(A, B)`                                           | mereological assertion *X is Y*                                    |
| `query(A, B)`                   | `Ops.query(A, B)` (new — spec O5)                          | interrogative speech act                                           |
| `swap(A, B)`                    | `Ops.swap(A, B)` (new — spec O5)                           | argument-position swap                                             |
| `bind(A, B)`                    | `Ops.bind(A, B)` (or `Ops.lower(mode='AND')` fallback)     | head-complement composition (PP, V + NP, …); see spec O6           |
| `scale(DEG, AP)`                | `Ops.scale(DEG, AP)` (or `Ops.lower(mode='AND')` fallback) | degree intensification; see spec O7 / O9                           |
| `project(A)` / single-symbol    | identity-with-typing-stamp                                 | terminal projection (`NP = project(N)` or `NP = N`); stamps LHS    |
| `identity(A)`                   | passthrough (returns `A` unchanged)                        | absorb single-position syntactic sugar (whitespace, punct word)    |
| `absorb(A, B)`                  | left-passthrough (returns `A`, discards `B`)               | absorb adjacent sugar (`B` learned to be punct/whitespace word)    |
| `*Reverse(Y)` / `*_inv(Y)`      | `Ops.<op>(Y, inverse=True)` (multi-return)                 | derived at grammar-load time — not authored separately (spec O10)  |

`Grammar.rules_upward` / `Grammar.rules_downward` hold the two rule
sets; `Grammar.rules` is their union.  Each `RuleDef` namedtuple carries
`(tier, canonical, arity, method_name, lhs, rhs_symbols)` — `method_name`
is the op name from the explicit-op / function-call RHS, or `'merge'`
for bare-symbol-sequence rules, or `'emit_head'` for the downward
`S -> C` projection.  The last two fields drive the typed-category
compatibility mask in the chart compose (see below).  Legacy flat
`<S>...</S>` directly under `<grammar>` (without the upward/downward
wrapper) is still accepted by `Grammar.configure()` for Python callers,
but XMLs shipped in this repo use the split form so the XSD can
validate them.

Reverse productions (the downward / inverse direction of every Layer-1
forward production) are *derived mechanically* at grammar-load time —
not authored as separate rules.  For each forward rule
`LHS = op(arg1, arg2)` the loader synthesizes `arg1, arg2 = opReverse(LHS)`
with the multi-return calling convention from
[specs/2026-04-24-lift-lower-bivector-design.md](specs/2026-04-24-lift-lower-bivector-design.md)
O10.

### Syntactic sugar consumption

Under the byte-level lex routed through BPE chunking, raw text positions
include plenty of *syntactic sugar* — runs of whitespace, single
punctuation marks (`,`, `.`, `;`, `:`, `"`, `'`, `(`, `)`), newlines.
After chunking these become their own word slots that the grammar must
either reduce away or carry along.

The legacy 17-rule grammar had no explicit absorption rule; the rule
predictor learned to abuse one of the existing binary ops (typically
`union(S, S)` or `swap(S, S)`) as a de-facto pass-through whenever one
operand was sugar. This was a bandaid that mixed semantically distinct
operations — propositional disjunction and "skip this comma" fired the
same rule with different probabilities depending on input.

The post-rolling-cursor grammar adds two explicit absorption rules:

- `identity(A)` — unary: returns `A` unchanged. Used when a single
  position is pure sugar (an isolated comma, a whitespace word) that
  should not contribute semantic content.
- `absorb(A, B)` — binary: returns `A`, discards `B`. Used when sugar
  attaches to a meaningful adjacent token (`"Hello,"` chunked as
  `[Hello, ,]` reduces via `absorb(Hello, ,)`).

Both rules are pure projections — no computation, no learnable
parameters in the projection itself. The rule predictor learns *when* to
fire them via the same softmax-over-rules logic as the semantic rules.
Initial logits should bias against `absorb` to avoid early-training
collapse where the parser learns to discard everything past the first
token; bias decays as the loss demonstrates the rule's actual usefulness.

### `data/grammar.cfg` dispatch (Step 6)

Step 6 of the lift / lower / bivector refactor introduced a text-based
rule table at `data/grammar.cfg` in the explicit-op form, alongside the
XML loader.  Format (line-oriented):

```
# comments start with '#'
[upward]
S = NP                        # PROJECT (single-category RHS)
S = lift(NP, VP)
NP = intersection(AP, NP)
S = not(S)
...

[downward]
C = emit_head(S)

[layer2]
# Post-hoc S-ops; not productions, no reverse pair.
S = true(S)
S = swap(S, S)
...
```

Sections are bracketed (`[upward]` / `[downward]` / `[layer2]`) and
default to `[upward]` when no header is seen.  Each `LHS = body` rule's
body is either a function call `op(arg1, arg2)`, a single category
(PROJECT), or `epsilon`.  Layer-2 rows are tracked separately on
`Grammar.rules_layer2` and stay out of `Grammar.rules` because they're
post-hoc transformations, not productions.

Loading is gated via `<WordSpace><language><grammarCfg>` in the XML
config — when set to a path (resolved against `ProjectPaths.PROJECT_DIR`),
`Grammar._ensure_configured()` calls `Grammar.load_from_cfg(...)` and
skips the legacy `<grammar>` block.  When unset, the XML grammar wins
(legacy path).  Both paths populate `Grammar.reverse_rules` with the
mechanically-derived Layer-2.5 reverse productions
`(args_tuple, reverse_op_name, (lhs,))`.

Dispatch itself is unchanged: `SyntacticLayer.project` looks up
`method_name` in `_RULE_METHODS` and calls the matching `<name>Forward`
on the layer.  The cfg path supplies the `method_name`; the layer
methods supply the Basis / mask / subspace context.  For callers that
want to invoke an op stateless (no Basis context),
`SyntacticLayer.dispatch_ops(op_name, *args, **kwargs)` resolves the
op name through `_OPS_METHODS` and calls the bare `Ops` callable
directly — useful for the explicit-op view from grammar.cfg without
going through layer plumbing.

The reverse op name follows the convention `<op>Reverse` for everything
except self-inverse ops (`not`) which keep their forward name; PROJECT
rules (single-category RHS) emit the explicit `projectReverse` reverse
name.  Multi-return reverses `Ops.liftReverseAll(Y[, W])` and
`Ops.lowerReverseAll(Y[, W])` return `(X1, X2)` tuples consistent with
the parent plan §Step 6 spec; the existing analytic single-arg forms
`Ops.liftReverse(result, right)` and `Ops.lowerReverse(result, right)`
are unchanged.

### Thought-Free Mode (Shamatha Speech)

`Grammar.thought_free` is a boolean flag (default `False`) settable
per-request by WikiOracle's `serve.py` before inference and reset in a
`finally` block.  Under the old three-tier grammar it suppressed C-level
`not`; after the S-merge its filtering effect is minimal pending
redesign.  See [Params.md](Params.md) for the XML configuration.

---

## Ops Reference

The grammar dispatches each rule to an `Ops.*` callable defined in
[`bin/Layers.py`](../bin/Layers.py) (class `Ops`).  This section is the
canonical reference for those callables: their current implementation,
their position on the layer-anchor map, and their status against the
target architecture (where every grammar op has a clean grammatical
interpretation).

### Layer-anchor map

Every grammar op should ultimately be implemented at one of four
anchor sites, reflecting where its semantics actually live:

| Anchor | Lives at | Description |
|---|---|---|
| **Sym** | SymbolicSpace | per-symbol algebra on `subspace.activation` (the bivector `[aP, aN]` and the codebook).  Operations that are intrinsically about symbols. |
| **SigmaPi** | `ConceptualSpace.pi` $\leftrightarrow$ `SymbolicSpace.sigma` round trip | level-crossing operations.  `lift` runs `forwardSigma`; `lower` runs `reverseSigma`; `intersection` / `union` round-trip through both. |
| **Per** | PerceptualSpace | percept-side operations (e.g. `chunk`).  Currently outside the grammar dispatcher. |
| **Def** | WordSpace | *definitional*: introduces a new symbol or asserts a relation in the WordSpace symbol table.  No tensor algebra; it edits the symbolic codebook / mereological relation table. |

The xAI / HI precondition is that **every** op has an explicit
grammatical interpretation, anchored at one of the four sites.  The
table below shows the mapping; the per-op subsections that follow
describe the current implementation and what changes when the op
migrates to its target anchor.

### Target $\leftrightarrow$ current $\leftrightarrow$ status table

| Op | Target | Current `Ops.*` body | Migration status |
|---|---|---|---|
| `true` | Sym | `Ops.positive(x) = relu(x)` | scalar primitive; needs Sym anchor for the "committed yes" half-space |
| `false` | Sym | `Ops.negative(x) = -relu(-x)` (or `relu(-x)` via `_OPS_METHODS`) | scalar primitive; same as `true` |
| `non` | Sym | `1 - |clamp(x, -1, 1)|` (bitonic) or `relu(x - threshold)` (monotonic) | scalar primitive; needs Sym anchor |
| `not` | Sym | `Ops.negation(x)` — bitonic sign flip; monotonic does paired-pole swap on bivector last dim | symbolic-side operation; **proposed: introduce a learnable NOT layer just after SigmaLayer** so Pi/Sigma stay positive-monotonic and `not` emits the negated word in the grammatical derivation |
| `conjunction` | Sym | `Ops.lower(x, y, mode='AND', kind=...)` (smooth: product; strict: min; radial: RadMin) | already routes through the dispatcher; symbolic-side per-symbol algebra |
| `disjunction` | Sym | `Ops.lift(x, y, mode='OR', kind=...)` (smooth: mean; strict: max; radial: RadMax) | already routes through the dispatcher; symbolic-side per-symbol algebra |
| `intersection` | SigmaPi | `Ops.lower(x, y, mode='AND')` (alias for `conjunction` in `_OPS_METHODS`) | currently identical to `conjunction`; **target: round-trip through `ConceptualSpace.pi.reverse → SymbolicSpace.sigma.forward`** so the result is meaningfully a level-crossing intersection of the witness regions, not just an `Ops.*` algebra |
| `union` | SigmaPi | `Ops.lift(x, y, mode='OR')` (alias for `disjunction`) | same as `intersection` — currently a pure-algebra alias; target is the SigmaPi round-trip form |
| `lift` | SigmaPi | `Ops.lift(X1, X2, mode, kind, inverse, monotonic)` — pure stateless dispatcher | target: `SymbolicSpace.sigma.forward(...)` (the SigmaLayer body); the cfg dispatcher could route through `sigma.forward(packed, binary=True)` to use the Step-5 STE top-2 selector |
| `lower` | SigmaPi | `Ops.lower(X1, X2, mode, kind, inverse, monotonic)` — pure stateless dispatcher | target: `ConceptualSpace.pi.forward(...)` (or its `reverse` for the inverse direction) |
| `equal` | Def | `Ops.equal(x, y) = part(x, y) * part(y, x)` (mutual parthood, vector or scalar) | currently computed as a tensor score; **target: a definitional op that introduces a new symbol into WordSpace asserting $x \equiv y$ as a recorded relation** |
| `part` | Def | `Ops.part(x, y) = x * (y / ‖y‖)` (vector) or clipped cosine in `[0, 1]` (scalar) | currently a tensor score; **target: a definitional op asserting `part(x, y)` as a stored mereological relation over the WordSpace codebook** |
| `what` / `where` / `when` | Sym | `Ops.{what,where,when}(x, nWhat, nWhere, nWhen)` — block selector zeroing the other two slots; identity when `x.ndim < 3` | symbolic-side; current impl already at Sym (operates on `subspace.activation`) |
| `query` | Mereological | `Ops.query` doesn't exist as an `Ops.*` body — handled in `compose()` via norm-drop detection; returns the left operand (preserved accumulator) | target: a mereological query over WordSpace's stored part / equal / overlap relations |
| `swap` | Sym | Sinkhorn soft-permutation matrix $P$ over 3 slots; $\mathrm{swap}(\ell, r) = (P \cdot [\ell, r, m]^\top)_0$ | symbolic-side; current impl is at Sym (acts on the per-position [where, when, what] slot tuple via a learned permutation) |
| `chunk` | Per | `Basis.disjunction(`$\ell, r$`, monotonic=True)` $\to \max(\ell, r)$ fallback; runs in the perceptual pre-pass | already at Per (called by `ChunkingLayer`); not currently in the grammar dispatch table |

The remainder of this section is the per-op specification: signature,
math, current location, and a note on what migrating to the target
anchor would change.

### Notation

Let $x, \ell, r \in \mathbb{R}^{B \times N}$ (activation mode) or
$\mathbb{R}^{B \times N \times D}$ (vector mode).  Let $\mathcal{B}$ be
the subspace basis (e.g. a bitonic `Basis`), which supplies pointwise
primitives `pos`, `negation`, `conjunction`, `disjunction`, `equal`,
`part`.  When $\mathcal{B}$ is absent the methods fall back to torch
elementwise primitives.  `mask` (optional concept-axis `Mask`) gates
the output by zeroing non-selected dimensions; it is omitted below for
clarity.

### Ternary operators (true / false / non) — Sym

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

### Negation (not) — Sym

$$
\mathrm{not}(x) = \mathcal{B}.\mathrm{negation}(x) = -x,
\qquad
\mathrm{not}^{-1} = -x
$$

Pure antipodal flip on the bitonic axis.  Self-inverse.

### Conjunction / disjunction — Sym

Both route through the unified `lift` / `lower` dispatcher (Step 1–2 of
[plans/2026-04-24-lift-lower-bivector-refactor.md](plans/2026-04-24-lift-lower-bivector-refactor.md)).
At the S tier the explicit-op grammar form `S = conjunction(S, S)`
resolves to `Ops.lower(S1, S2, mode='AND')` and `S = disjunction(S, S)`
resolves to `Ops.lift(S1, S2, mode='OR')`:

$$
\mathrm{conjunction}(\ell, r) = \mathcal{B}.\mathrm{lower}(\ell, r,\ \mathrm{mode}{=}\text{'AND'})
                              \ \xrightarrow[\text{kind='strict'}]{}\ \min(\ell, r)
$$

$$
\mathrm{disjunction}(\ell, r) = \mathcal{B}.\mathrm{lift}(\ell, r,\ \mathrm{mode}{=}\text{'OR'})
                              \ \xrightarrow[\text{kind='strict'}]{}\ \max(\ell, r)
$$

The `kind` parameter selects the forward point body: `'strict'`
(default — lattice min/max), `'smooth'` (elementwise product / mean,
the differentiable form used by `Pi`/`SigmaLayer.forward`), or
`'radial'` (RadMin / RadMax — same-sign min/max magnitude with zero
passthrough; equivalent to the historical `monotonic=False` body of
`Basis.conjunction` / `Basis.disjunction`).  The legacy thin-forwarder
methods `Basis.conjunction(x, y, monotonic=...)` /
`Basis.disjunction(x, y, monotonic=...)` are preserved as deprecated
aliases and route through the new dispatcher (`monotonic=True` →
`kind='strict'`; `monotonic=False` → `kind='radial'`) so existing
callers see bit-exact output.

Sentence-level fuzzy $\wedge$ and $\vee$ on bitonic activations.  Both
are lossy at the sentence level (no inverse registered for the S-tier
usage), though the same primitives at C-tier subspace level retain
Basis-level inverses via `Ops.conjunctionReverse` /
`Ops.disjunctionReverse` (codebook-search recovery).

### Intersection / union — SigmaPi (currently aliases `lower`/`lift`)

`intersection` and `union` are aliases for `conjunction` / `disjunction`
under the explicit-op grammar form, dispatching the same unified call:

$$
\mathrm{intersection}(\ell, r) = \mathcal{B}.\mathrm{lower}(\ell, r,\ \mathrm{mode}{=}\text{'AND'})
\qquad
\mathrm{union}(\ell, r) = \mathcal{B}.\mathrm{lift}(\ell, r,\ \mathrm{mode}{=}\text{'OR'})
$$

Inverses are the codebook-search recoveries on $r$ (supplied by `Basis`
which carries the codebook $W$):

$$
\mathrm{intersection}^{-1}(y, r) = \mathcal{B}.\mathrm{lower}(y, r,\ \mathrm{mode}{=}\text{'AND'},\ \mathrm{inverse}{=}\mathrm{True})
\qquad
\mathrm{union}^{-1}(y, r) = \mathcal{B}.\mathrm{lift}(y, r,\ \mathrm{mode}{=}\text{'OR'},\ \mathrm{inverse}{=}\mathrm{True})
$$

For `mode='NOT'` the inverse is identical to the forward (self-inverse
pole flip / sign flip).

### Part — Def (currently a tensor score)

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

### Equal — Def (currently a cosine-gated projection)

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

### Slot selectors (what / where / when) — Sym

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

### Query — Mereological (currently accumulator preservation)

$$
\mathrm{query}(\ell, r) = \ell
$$

The query marker is pushed onto WordSubSpace at the accumulation point
by `compose()` when it detects norm-drop (see below), not by
`queryForward` itself.  When the parse tree is re-evaluated downstream,
$\mathrm{query}$ returns the preserved accumulator (the left operand).
Lossy.

### Swap (Sinkhorn soft permutation) — Sym

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

### Lift / lower — SigmaPi (mode-dispatch ops)

`Ops.lift` and `Ops.lower` are unified mode-dispatchers.  Default
modes are `lift` $\to$ `'OR'` (synthesis, $\vee$) and `lower` $\to$ `'AND'` (analysis,
$\wedge$); each accepts `kind` $\in$ `{'strict', 'smooth', 'radial'}` selecting the
forward body, and `inverse=True` for the reverse direction.

**Forward bodies (point):**

| | `mode='AND'` (`lower`) | `mode='OR'` (`lift`) |
|---|---|---|
| `kind='strict'` | $\min(X_1, X_2)$ | $\max(X_1, X_2)$ |
| `kind='smooth'` | $X_1 \cdot X_2$ (elementwise product) | $(X_1 + X_2) / 2$ (mean) |
| `kind='radial'` | RadMin (same-sign min magnitude, zero-collapse) | RadMax (same-sign max magnitude, zero-passthrough) |

**Region forms (operand is a tuple $(\ell, u)$):** `lower(mode='AND')`
returns $(\max(\ell_1, \ell_2), \min(u_1, u_2))$; `lift(mode='OR')`
returns $(\min(\ell_1, \ell_2), \max(u_1, u_2))$ — independent of
`kind`.

**Cross-mode delegation:** `lift(mode='AND')` forwards to
`lower(mode='AND')` for symmetry; `lower(mode='OR')` forwards to
`lift(mode='OR')`.  `mode='NOT'` is `Ops.negation(X1, monotonic=...)`
on either dispatcher (self-inverse).

**Reverse / inverse:**

- `inverse=True, mode='AND' or 'OR'` requires a codebook `W` and
  raises `NotImplementedError` if unsupplied — callers must use the
  `Basis.*` flavor, or `Ops.conjunctionReverse(result, y, W)` /
  `Ops.disjunctionReverse(result, y, W)` directly.
- `Ops.lowerReverseAll(Y, W, monotonic)` and
  `Ops.liftReverseAll(Y, W, monotonic)` return the multi-operand pair
  `(recovered_left, recovered_right)` per the parent plan §Step 6
  Layer-2.5 grammar convention.
- The single-operand analytic forms `Ops.liftReverse(result, right)` =
  $\mathrm{result} / (\mathrm{right} + \varepsilon)$ and
  `Ops.lowerReverse(result, right)` = $2 \cdot \mathrm{result} -
  \mathrm{right}$ are preserved as legacy callers' inverse of the old
  `Ops.lift` $= \odot$ and `Ops.lower = mean` bodies.  They warn-only and
  are permanent.

**Layer integration.** The cfg-driven grammar dispatch invokes
`Ops.lift` / `Ops.lower` directly (option (a) per the Step 6.5
closure: explicit two-operand calls).  The learnable equivalents live
on `SymbolicSpace.sigma.forward` / `ConceptualSpace.pi.forward` which
own their own log-domain (Pi) and atanh-domain (Sigma) matmul bodies.
The bridge between dispatch-time algebra and learnable layers is the
`binary=True` flag on `PiLayer.forward` / `SigmaLayer.forward` (Step
5): it pre-applies `Ops.top2_select_ste` to hard-select the top-2
input operands before the layer body runs.

### Chunk — Per (invertible max-union; pre-pass)

$$
\mathrm{chunk}(\ell, r) = \mathcal{B}.\mathrm{disjunction}(\ell, r,\ \mathrm{monotonic}{=}\mathrm{True})
\ \xrightarrow[\text{fallback}]{}\ \max(\ell, r)
$$

Monotonic max-union.  Inverse via
$\mathcal{B}.\mathrm{disjunction\_inverse}(y, r)$.  Not registered in the
default S-tier rule list; used by `ChunkingLayer` in the perceptual
pre-pass.  This is the only Per-anchored op currently shipping.

### Reverse / inverse ops

These are the analytic and codebook-search inverses paired with the
forward ops above.  None are dispatched by the cfg grammar directly —
they are invoked by `Basis.*` methods or by the parent plan's Layer-
2.5 grammar reverse convention.

| Op | Behavior |
|---|---|
| `Ops.negationReverse(x, monotonic)` | self-inverse — calls `Ops.negation(x, monotonic)` again |
| `Ops.conjunctionReverse(result, y, W, monotonic)` | codebook-search recovery — find $cb_i$ such that $\mathrm{conjunction}(cb_i, cb_j) \approx \mathrm{result}$; returns recovered left operand |
| `Ops.disjunctionReverse(result, y, W, monotonic)` | dual of `conjunctionReverse`; recovers left operand of disjunction |
| `Ops.liftReverse(result, right)` | analytic single-operand inverse of the legacy `Ops.lift` $= \odot$ body: $\mathrm{result} / (\mathrm{right} + \varepsilon)$.  Permanent, warn-only. |
| `Ops.liftReverseAll(Y, W, monotonic)` | multi-return Step-6 form: $(\mathrm{recovered\_left}, Y)$ via codebook search; falls back to $(Y, Y)$ when $W$ is unavailable |
| `Ops.lowerReverse(result, right)` | analytic inverse of legacy `Ops.lower = mean`: $2 \cdot \mathrm{result} - \mathrm{right}$.  Permanent, warn-only. |
| `Ops.lowerReverseAll(Y, W, monotonic)` | dual of `liftReverseAll` — multi-return form for the Step 6 grammar reverse pass |

The `_binary_op_inverse_impl(result, W, op, monotonic)` private helper
backs both `conjunctionReverse` and `disjunctionReverse`.  It searches
the codebook $W$ for the pair $(cb_i, cb_j)$ whose forward $op$ best
matches `result`, returning $cb_i$.

### Mereological suite — see [Mereology.md](Mereology.md)

The mereology suite (`part`, `whole`, `equal`, `overlap`, `underlap`,
`boundary`, `copart`) lives in `Ops.*` but the canonical specification
of its semantics is [Mereology.md](Mereology.md).  Each member has a
vector form (default) and a scalar form (`scalar=True`):

| Op | Vector form | Scalar form |
|---|---|---|
| `part(x, y)` | $x \cdot (y / \|y\|)$ | clipped cosine $\max(0, x \cdot y) / (\|x\|\|y\|) \in [0, 1]$ |
| `whole(x, y)` | $(1 - x) \cdot (y / \|y\|)$ | $\mathrm{part}(y, x, \mathrm{scalar}{=}\mathrm{True})$ |
| `equal(x, y)` | $\mathrm{part}(x, y) \cdot \mathrm{part}(y, x)$ | mutual parthood $\in [0, 1]$ — $0$ disjoint, $\in (0, 1)$ overlap, $1$ identity |
| `overlap(x, y)` | $\min(\mathrm{part}(x, y), \mathrm{part}(y, x))$ | boolean: $0 < \mathrm{equal}(\ldots) < 1$ |
| `underlap(x, y)` | $\min(\mathrm{whole}(x, y), \mathrm{whole}(y, x))$ | boolean: $\mathrm{equal}(\ldots) = 0$ |
| `boundary(x, y)` | $\lvert \mathrm{part}(x, y) - \mathrm{part}(y, x) \rvert$ | always $0$ under the symmetric clipped-cosine score |
| `copart(x, y)` | $y - x$ | $1 - \mathrm{part}(x, y, \mathrm{scalar}{=}\mathrm{True})$ |

Empty-set conventions (parthood scalar form):
$\mathrm{part}(\emptyset, y) = 1$,
$\mathrm{part}(x, \emptyset) = 0$,
$\mathrm{part}(\emptyset, \emptyset) = 1$.

`part` and `equal` are also surfaced through the grammar dispatch
table; the rest of the suite is utility-level (called by `Basis.*` and
the TruthLayer derivation pipeline).  The target architecture moves
`part` and `equal` from tensor scoring to **Def** anchors that
introduce a recorded relation in WordSpace; see the table at the top
of this section.

### Pointwise primitives

These are the small utility ops with no current grammatical role —
they are called by the layer / loss / encoding code paths and have no
dispatch entry.  Listed for completeness so the inventory is closed.

**Constants:** `Ops.true() = 1`, `Ops.false() = -1`,
`Ops.unknown() = 0`.

**Element-wise:**

| Op | Body |
|---|---|
| `Ops.positive(x)` | $\mathrm{relu}(x)$ |
| `Ops.negative(x)` | $-\mathrm{relu}(-x)$ |
| `Ops.neutral(x)` | $1 - \lvert x \rvert$ |
| `Ops.pos(x)` | $\mathrm{relu}(x)$ — alias used by the bivector path |
| `Ops.sign(v)` | signum with $\mathrm{sgn}(0) = 1$ (not $0$) |
| `Ops.saturate(x)` | $\mathrm{clamp}(x, -1, 1)$, NaN $\to 0$ |
| `Ops.threshold(x, t)` | zero entries with $\lvert x \rvert < t$ |
| `Ops.complement(x)` | $\mathrm{sign}(x) - x$ |
| `Ops.convertSensation(x)` | $2x - 1$ — $[0, 1] \to [-1, 1]$ remap |
| `Ops.minMag(x_1, x_2)` | $x_i$ with smaller $\lvert x_i \rvert$ |
| `Ops.maxMag(x_1, x_2)` | $x_i$ with larger $\lvert x_i \rvert$ |
| `Ops.error(x_1, x_2)` | $\| x_1 - x_2 \|_2$ — scalar L2 |
| `Ops.norm(x)` | last-dim L2 norm |
| `Ops.distance(x, y, monotonic, dim)` | $(1 - \cos(x, y)) / 2$ (bitonic) or volume-weighted L2 (monotonic) |

**Predicates (return Boolean):**

| Op | Body |
|---|---|
| `Ops.isActive(x, t)` | $\lvert x \rvert \ge t$ |
| `Ops.isEqual(x_1, x_2)` | `torch.equal` strict equality |
| `Ops.isReducer(x_1, x_2)` | $\| x_2 - x_1 \|_1 < \| x_2 \|_1$ — scalar Bool |

**Selection (Step-5 STE bridge):** `Ops.top2_select_ste(x, dim=-1)`
selects the top-2 entries of $x$ by $\lvert x \rvert$, zeroing the
rest.  Backward is straight-through.  Used by
`PiLayer.forward(x, binary=True)` and `SigmaLayer.forward(x,
binary=True)` to specialize the layer body to per-rule binary
application.

**Private helpers** (referenced by the public ops above):
`_radmin(x, y)`, `_radmax(x, y)` — the same-sign min / max magnitude
bodies used by `kind='radial'`; `_binary_op_inverse_impl(result, W,
op, monotonic)` — the codebook-search backbone for
`conjunctionReverse` / `disjunctionReverse`; `_as_region(x)` —
promotes a point to a degenerate region $(\ell, u) = (x, x)$ for the
region-form bodies.

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
Inputs are `[B, N]` activations (2D mode) or `[B, N, D]` vectors
(3D mode).  (The "bivector" name is reserved in the code for the
2-component `[..., :2]` slice of a symbol codebook row, not the full
`D`-wide vector.)

`reverse_project` is the dual, used during `decompose` to invert the
registered reversible rules.

---

## Composition: `compose(data, subspace, grammar, target_count=None)`

`SyntacticLayer.compose()` is the unified two-phase driver.  It
dispatches on input rank:

- `data.ndim == 2`  -->  `_compose_activation` (2D activation mode)
- `data.ndim == 3`  -->  `_compose_vector`     (3D vector mode,
  with optional pairwise reduction via `_compose_to_target`)

Returns `(composed, svo_or_None)`.  `svo` is a
`(subject, verb, object)` tuple of `[B, 1, D]` tensors when the
chart-compose path (see below) finds a canonical `S -> S VO` firing
and its matching `VO -> V O`; otherwise `None`.

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

Before Phase 1 begins, codebook indices of the original leaves are
snapshotted (so the ledger reflects the pre-`not` input exactly):

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

## Chart-like Compose (typed SVO)

Selected by `<WordSpace><chartCompose>true</chartCompose></WordSpace>`.
The cascade-based `_compose_vector` above builds a strictly left-
associative tree `(((leaf0, leaf1), leaf2), ...)`; it cannot produce
`(S, (V, O))` from an `N V N` sentence because step 1 is forced to
merge `leaf[0]` with `leaf[1]` (an `N V` pair) and no rule in a typed
grammar fires on that shape.  Chart compose replaces the cascade with
per-step pair selection, so the model can reduce `V O` first and then
`S VO` second.

### Pair-scorer + per-step argmax

At each depth step:

1. A pair-scorer MLP takes the current rule-prediction hidden state
   plus `(left, right)` feature slices and produces a softmax over the
   `N_alive - 1` adjacent live-leaf pairs.  Input width is
   `hidden_dim + 2 * feature_dim`; `feature_dim` is a first-class
   `SyntacticLayer.__init__` kwarg so the pair-scorer lines up with the
   actual leaf dim at compose time (not `n_slots`, which tests happen
   to set equal to `D`).  Note: until `MentalModel` pipes the
   rule-prediction hidden through, `_compose_vector_chart` passes an
   all-zero `[B, hidden_dim]` tensor (see `Language.py:1354`); the
   pair scorer therefore conditions only on `(left, right)` features
   today.
2. The rule head emits the usual per-depth softmax over composable
   rules (upward rules only; downward is emit-time, not chart-time).
3. A **compatibility mask** `compat[B, P, R]` zeroes out `(pair, rule)`
   combinations whose typed LHS/RHS categories don't match the pair's
   slot categories.  Function-call rules (legacy, `rhs_symbols = None`)
   are always compatible.  Category `0` (`'?'`) is a wildcard, so
   unseeded leaves can match any typed rule during warmup.
4. The joint `pair_probs * rule_probs * compat` is renormalised.  In
   training the merge output is the soft mixture; at eval the argmax
   pair / rule pick the concrete merge site.
5. The merged vector replaces the left slot in an out-of-place
   `torch.where`-style update (inplace mutation on the live tensor
   would break autograd because `live = data.clone()` carries
   `grad_fn`).  The right slot is marked dead via the alive mask; the
   merged slot's category becomes the rule's LHS.

The reduction runs for at most `N - 1` steps, so `N` live leaves
collapse to one in `N - 1` merges.

### Category tensor + POS seeding

Alongside `data: [B, N, D]`, the chart path carries
`category: [B, N]` of category IDs (the `Grammar.rules`-derived table
is stored on `SyntacticLayer._category_names` / `_category_index`,
with `'?'` at index 0 as the wildcard).  `_seed_category` lets
upstream code pre-populate slot categories.

Category seeding is currently a no-op stub; the wildcard `?` keeps the
compat mask permissive.  Upstream code may call `_seed_category`
directly to bias the chart toward specific categories when a
domain-specific tagger is available.

### Derivation trace

`SyntacticLayer._derivation_trace` is a per-batch list (one list per
row in the batch) of 5-tuples
`(rule_id, left_slot, right_slot, merged_slot, merged_category)`
appended on every chart merge.  The trace is the only surface that
downstream readers (SVO extractor, future reconstruct templates) rely
on; the `live` tensor is already compacted by the time compose
returns.

### Grammar-derived SVO

`SyntacticLayer._extract_svo_from_trace` walks each batch row's trace
after chart compose finishes:

1. Find the last `S -> S VO` entry (outermost reduction).
2. Find the matching `VO -> V O` entry whose `merged_slot` equals
   the outer rule's right arg.
3. Pull `subject` from the outer's left slot, `verb` from the inner's
   left, `object` from the inner's right.  Each is a `[1, D]` slice of
   the original pre-compose `data`.

Rows that don't contain the canonical pair of firings get a zero `[1, D]`
placeholder; if every row in a batch fails, `last_svo` stays `None` so
`truth_modulated_loss` treats it as "no score".  There is no positional
fallback — the earlier heuristic that labelled positions `0/1/2` as
`S/V/O` is gone.  Consumers read `SyntacticLayer.last_svo` exactly as
before.

## Downward Head Emission (`S -> C`)

Selected by
`<WordSpace><downwardGeneration>true</downwardGeneration></WordSpace>`.
After the upward parse reduces `N` leaves to a single root vector,
`MentalModel.forward` takes that root (`sym_vectors[:, 0, :]`) and
calls `WordSpace.reconstruct(state, inputSpace)`, which runs the
downward `S -> C` rule as a one-shot projection onto a codebook.

### `SyntacticLayer.emit_head(state, codebook) -> (best_idx, contained, residual)`

Projection-coefficient scoring, not cosine:

$$
W_{\text{norm}} = W / \lVert W\rVert_{2,\text{row-wise}}
\qquad
\text{scores} = \mathrm{clamp}_{\ge 0}(\text{state} \cdot W_{\text{norm}}^\top)
$$

The per-row argmax is `best_idx`.  Because `state` is *not* L2-
normalised (only the codebook rows are), the score is "how much of
atom $k$ lives in the state" rather than a pure angle.  The resulting
`contained = scores[:, k] * W_{\text{norm}}[k]` is the slice of the
atom actually contained in the state; `residual = state - contained` is
the leftover meaning a future expansion step (NP / VP templates) could
consume.

### `WordSpace.reconstruct(state, codebook_space)`

Any space whose `.subspace.basis` exposes a `getW() -> [V, D]` works.
`InputSpace` (word embedding) is the intended source — projecting the
deep state onto word-vectors picks the vocabulary entry that best
matches the sentence's composed meaning, and `.wv.index_to_key` decodes
the head index back to a token.  `SymbolicSpace` (internal atom
codebook) is also valid once a codebook is populated there.  When the
codebook is unwired (passthrough `Codebook` with `getW() == None`),
`reconstruct` returns a trivial `heads=[0]*B` so the loss path never
crashes on a `None` codebook.

Returns `{heads: list[int] (len B), contained: [B, D], residual: [B, D],
state: [B, D]}`.  `MentalModel.forward` stashes `heads` as
`self._predicted_head` — a per-batch list of vocabulary indices — so a
training loss can compare against a supervised head token, and so
`bin/interact_head.py` (interactive smoke test) can print the decoded
word after each forward pass.

### Stable hooks already wired

- `SyntacticLayer.last_svo: Optional[Tuple[Tensor, Tensor, Tensor]]`
- `SyntacticLayer.lifting_layer: LiftingLayer` (created in
  `init_lifting`, called from `WordSpace._build_syntactic_layer`)
- `MentalModel.forward` reads both, invokes
  `truth_layer.universality(s, v, o, lifting_layer, symbolicSpace)`,
  and stores `self._universality_score`; `truth_modulated_loss`
  integrates the score into the training loss.
- `MentalModel._predicted_head: Optional[list[int]]` set by the
  downward pass above.

The historical design narrative (why chart compose was needed, why a
cheaper hybrid doesn't work, curriculum plan for training) lives in the
"Design History: Learned SVO Plan" appendix at the end of this document,
and the implementation tracker at
`doc/plans/2026-04-20-LearnedSVO-integration.md`.

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

---

## Design History: Learned SVO Plan

This appendix preserves the original design narrative that justified
moving from a positional SVO tap to grammar-derived SVO roles. The plan
below was authored before the chart-compose implementation landed;
Phases A-D have since shipped (see the "Chart-like Compose (typed SVO)"
section earlier in this document and the tracker at
`doc/plans/2026-04-20-LearnedSVO-integration.md`). Kept here for
archaeology and as a reference for any future extension of the same
approach.

### Learned SVO Role Identification from the Grammar

This section specifies what it would take to drive
`TruthLayer.universality` from grammar-derived Subject/Verb/Object
roles rather than the positional heuristic previously in use.

`TruthLayer.universality(subject, verb, object, lifting_layer, symbolic_space)`
implements the Golden Rule: it scores luminosity change under S/O reversal,
so `K(X,Y) + K(Y,X)` illuminates more than `K(X,Y)` alone for kind actions.
The method takes `(S, V, O)` concept tensors as inputs; the question is where
those three tensors come from.

### Current state (at the time of writing)

SVO was produced by a positional tap inside
`SyntacticLayer._compose_vector` in `bin/Language.py`: the first three active
leaf positions of a batch row were labelled subject/verb/object when every
row had at least three active leaves. For the canonical three-token
transitive corpus in `test/test_universality.py`
(`"the teacher helped the student"` without determiners) this produced
the right roles. It was wrong for realistic inputs: determiners shift the
roles, adjectives insert modifiers, word-order variation breaks it entirely,
and embedded clauses are invisible.

The plumbing downstream of `last_svo` was already correct for a learned tap
to drop into:

- `SyntacticLayer.last_svo: Optional[Tuple[Tensor, Tensor, Tensor]]`
- `SyntacticLayer.lifting_layer: LiftingLayer` (instantiated in `init_lifting`,
  called from `WordSpace._build_syntactic_layer`)
- `MentalModel.forward` reads both, calls `truth_layer.universality(...)`,
  and exposes the result via `self._universality_score`
- `truth_modulated_loss(universality_score=...)` folds the score into the
  training loss

Only the producer of `last_svo` needed to change.

### Target grammar shape

The binary grammar should be able to express:

    S  -> S VO
    VO -> V O
    V  -> v(V)      -- terminal category verb
    O  -> n(O)      -- terminal category noun

so that a parse of a three-content-word sentence produces a derivation tree
whose outer rule is `S -> S VO`, and whose `VO` child comes from `VO -> V O`.
The subject vector is arg-0 of the outer rule; the verb vector is arg-0 of
the inner rule; the object vector is arg-1 of the inner rule.

Longer inputs extend naturally:

    S   -> NP VP
    NP  -> DET N | N
    VP  -> V NP
    NP' -> DET N | ADJ N | DET ADJ N

and an SVO walker reads (subject = arg-0 of S, verb = head of VP, object =
head of VP's NP).

### Phase A - Grammar metadata (low risk, ~60 LoC)

#### A.1 Extend `RuleDef`

```python
RuleDef = namedtuple(
    'RuleDef',
    ['tier', 'canonical', 'arity', 'method_name', 'lhs', 'rhs_symbols'],
)
```

- `lhs`: nonterminal name this rule reduces to (`'S'`, `'VO'`, `'NP'`, ...).
- `rhs_symbols`: tuple of nonterminal / terminal category names for each
  RHS slot (e.g. `('V', 'O')` for `VO -> V O`).

Existing rules default `lhs='S'`, `rhs_symbols=('S', ...)` for backwards
compatibility with the current single-tier S grammar.

#### A.2 Multi-LHS `configure()`

`Grammar.configure()` hardcodes `for lhs in ('S',)` (see `bin/Language.py`
around the `configure` method). Replace with iteration over every key in
the XML `<grammar>` block, so each key becomes a distinct nonterminal
with its own rule list.

#### A.3 RHS parser changes

`_parse_rule` currently parses function-call syntax (`lift(S, S)`,
`not(S)`). Add a bare-symbol-sequence form for typed rules:

    <S>S VO</S>           -- rhs_symbols=('S', 'VO'), method='merge'
    <VO>V O</VO>          -- rhs_symbols=('V', 'O'), method='merge'

Branch on whether the RHS contains `(` / `)`. The function-call form stays
for existing ops; the bare-symbol form is the new typed form.

#### A.4 XML/XSD schema

`MentalModel.xml` and its XSD must allow non-S nonterminals as child
elements of `<grammar>`. Add each new nonterminal tag as an optional
element in the XSD. No migration needed for existing configs because
`<S>...</S>` stays valid.

### Phase B - Compose restructure (high risk, ~120 LoC)

#### B.1 Problem: left-associative cascade can't produce `(S, (V, O))`

`_compose_vector` in `bin/Language.py` is a strict left-associative cascade:

    composed = leaf[0]
    for d:
        composed = rule_d(composed, leaf[d+1])

This bracketing is `(((leaf0, leaf1), leaf2), ...)`. To realise `S -> S VO`
where `VO` is itself `V O`, we need `(leaf1, leaf2)` to merge first into VO,
then `(leaf0, VO)` into S. Left-associative cascade cannot express this.
`_compose_to_target` has the same limitation - it pairs (0,1), (2,3),
etc. leftmost-first.

#### B.2 Chart-like pair selection

Replace the cascade with per-step pair selection:

- At each composition step the rule head picks, differentiably, WHICH two
  adjacent leaves to merge plus WHICH rule to apply.
- A pair-scorer MLP takes the current hidden state plus a pair context
  `(left, right)` and outputs a softmax over `(N-1)` adjacent pairs.
- The selected pair is merged; the resulting slot replaces the pair in
  the live-leaf array. Total steps: `N-1`.
- Maintain a per-batch live-leaf count and an alive-mask `[B, N_max]`.

#### B.3 Risks

- O(N^2) per depth step at train time (soft pair mixture over pairs x
  rules) vs O(1) in the current code. Acceptable for short inputs
  (N <= 16); may need chunking for longer.
- Hard argmax at eval, soft at train; convergence needs care.
- `decompose()` needs a matching inverse that reads the pair-selection trace.
- `subspace.add_word()` currently records a single position per rule firing.
  Pair-selection records `(pos_left, pos_right)` and a merged-slot position;
  `WordEncoding.encode(...)` already has `leaf1/leaf2/leaf3` fields for this.

#### B.4 Why a cheaper hybrid doesn't work

One might keep the left-assoc cascade and let the rule head pick rules
whose LHS is only reachable after the inner merge has happened. For `N V N`:

- Step 1 wants `V N -> VO` (inner first).
- Step 2 wants `N VO -> S` (outer second).

The left-assoc cascade forces step 1 to merge `leaf[0]` with `leaf[1]`
(i.e. `N V` first), which is `N V` - not a rule in the target grammar.
No rule fires, compose stalls. Left-assoc is unsalvageable for standard
SVO. Phase B is required.

### Phase C - Category tensor alongside activation (~60 LoC)

#### C.1 Per-slot category

Augment the data tensor `[B, N, D]` with a category tensor
`category: [B, N]` where `category[b, n]` is the nonterminal / terminal
category ID at that slot (or `-1` for padding). This rides alongside
activation through compose.

#### C.2 Initialisation from POS

Two options for seeding category at input time:

- **(a) External tagger.** Run an external POS tagger at data-prep time,
  map tags to our lexical categories (`N`, `V`, `ADJ`, `ADV`, `DET`, ...),
  store category indices alongside token indices. Clean; requires plumbing
  through `InputSpace` and pulling a tagger dependency back in.
- **(b) Codebook-derived POS.** K-means on the `WordVectors` codebook to
  produce K lexical clusters; at runtime, look up each leaf's cluster as
  its category. Cheaper, co-trained, no external dependency. Quality
  depends on embedding geometry.

(b) is now the preferred path — NLTK has been removed from the project.

#### C.3 Rule-compatibility masking

The rule head emits `rule_probs: [B, max_depth, num_rules]`. Extend to a
compatibility mask:

- For each candidate rule `r`, its `rhs_symbols` specifies the nonterminals
  the RHS must hold.
- At each merge step, compute `compat[b, r] = 1` iff the categories of the
  two candidate children match `r.rhs_symbols`, else `0`.
- Multiply into `rule_probs`, then renormalise. Incompatible rules drop out.
- On merge, set the merged slot's category to `r.lhs`.

#### C.4 Gradient / training dynamics

A hard compat mask kills gradient through incompatible rules - semantically
correct (those rules cannot fire) but brittle early in training if category
assignment is wrong. Mitigations:

- Small epsilon on incompatible rules early; anneal to zero over training.
- Gumbel-softmax on the merged-slot category assignment to keep gradients
  flowing through category propagation; anneal temperature high-to-low.
- A designated always-compatible fallback rule (e.g. `chunk`) so compose
  never stalls on unknown pairs.

### Phase D - SVO extraction hook (~30 LoC)

Once C is working, SVO falls out of the derivation trace.

#### D.1 Record the trace

In `_compose_vector`, each rule firing produces a tuple
`(rule_id, left_slot, right_slot, merged_slot)`. Accumulate per-batch into
`self._derivation_trace: List[List[Tuple[...]]]` over depth.

#### D.2 Walk the trace to SVO

After compose completes, for each batch row:

1. Find the outermost `S -> S VO` firing (argmax rule at the final step,
   constrained to `S`-LHS rules).
2. Read its left arg as subject (slot pointing to an `S`-category leaf or
   subtree - the canonical S subject is a noun phrase whose head is the
   subject noun).
3. Find the `VO -> V O` firing that produced the right arg of step 1.
4. Read its left arg as verb, its right arg as object.

Each of S/V/O is the concept vector at the respective slot in `data`
(or the aggregate of the subtree slots if we want phrase heads rather than
single-word heads).

#### D.3 Batch-level variability

Per-batch traces are natural (rule firings are already per-batch). SVO
extraction runs per batch and produces a per-batch mask of "has SVO"; rows
without the canonical shape leave their `last_svo` entry as None. The
`MentalModel` hook already handles a global `None`; extending to a batch
mask means `truth_layer.universality(s, v, o, ..., mask=...)` filters out
rows without valid SVO before computing luminosity deltas.

### Phase E - Training dynamics

#### E.1 Soft-mixture / hard-gate interaction

Phase B's soft pair mixture and Phase C's hard category gate interact:
- Rule output is a soft mixture over compatible rules.
- Merge output is a soft mixture over pairs.
- Merged-slot category is a categorical argmax of rule LHS (hard).

If hard category assignment is wrong, subsequent compat masks are wrong.
Gumbel-softmax on the category assignment lets gradients backpropagate
through the "wrong" choice and correct it.

#### E.2 Curriculum

- Start with short canonical inputs (3-content-word SVO, matching the
  test corpus) where the correct parse is unambiguous.
- Extend to 5-token `DET N V DET N` once short inputs produce stable SVO.
- Add modifiers (`ADJ`, `ADV`) and embedded clauses last.

#### E.3 Universality loss weighting

`architecture.UniversalityWeight` currently defaults to 0.1. Once SVO is
learned rather than heuristic, raise toward 0.3-0.5 - the Golden Rule
signal is only stable when SVO extraction is reliable, and only then
should it dominate loss.

### Phase F - Test suite

#### F.1 New unit tests

- `test_ruledef_typed.py`: `RuleDef` stores `lhs` / `rhs_symbols`; grammar
  exposes them; XML round-trip preserves them.
- `test_compose_chart.py`: chart-like compose reduces `N` leaves to `1` in
  `N-1` steps, preserves tensor dims, handles alive-mask.
- `test_category_propagation.py`: merging `(N, V)` under `VP -> V N` yields
  a slot with category `VP`; incompatible pairs zero the rule prob.

#### F.2 Update `test/test_universality.py`

`_get_svo_and_luminosity` currently returns `(None, None)` with a comment
noting that the C-tier ternary-lift path is gone. Once Phase D is in place,
replace it with a derivation-trace walk that reads SVO from the model and
passes through `truth_layer.universality(...)`. The `xfail` markers stay:
untrained models will not score `kind > unkind`.

### Cost estimate

| Phase | LoC | Risk   | Depends on |
|-------|-----|--------|------------|
| A: Grammar metadata        | ~60  | low    | -          |
| B: Chart-like compose      | ~120 | **high** | A        |
| C: Category tensor         | ~60  | medium | A, B       |
| D: SVO hook                | ~30  | low    | C          |
| E: Training dynamics       | tuning | medium | D        |
| F: Tests                   | ~80  | low    | A, B, C, D |
| **Total**                  | ~350 |        |            |

Phase B was the load-bearing risk. Allow at least a week for the compose
restructure; everything else layers cleanly once B is stable.

### Interim: positional tap (the pre-chart stopgap)

Until the learned approach landed, `SyntacticLayer._compose_vector` used a
positional heuristic: any batch row with at least three active leaves
labels positions 0/1/2 as subject/verb/object. This fired for the
`test_universality.py` canonical corpus and silently skipped longer inputs
(`last_svo` stayed `None`, universality stayed `None`, the loss term
contributed zero).

Known limits of the positional tap:

- Wrong SVO for five-token sentences with determiners (picks the determiner
  as "subject").
- No role fidelity for non-canonical word order.
- No support for embedded clauses (only the top-level SVO is visible).

All three are addressed by Phases A-D.

### Appendix - hooks that Phase D builds on (already wired)

- `SyntacticLayer.last_svo: Optional[Tuple[Tensor, Tensor, Tensor]]`
  (set in `compose`, populated in `_compose_vector`).
- `SyntacticLayer.lifting_layer: LiftingLayer` (created in `init_lifting`,
  called from `WordSpace._build_syntactic_layer`).
- `MentalModel.forward` reads both, invokes
  `truth_layer.universality(s, v, o, lifting_layer, symbolicSpace)`,
  and stores `self._universality_score`.
- `truth_modulated_loss(universality_score=..., universality_weight=...)`
  integrates the score into the training loss.

These four interfaces stay stable when learned SVO replaces positional SVO.
The only change is how `last_svo` is produced.
