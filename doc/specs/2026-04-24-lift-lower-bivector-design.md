# Spec: Lift / Lower / Bivector Refactor — Design Resolutions

**Date.** 2026-04-24
**Owner.** Alec
**Status.** Draft.  Resolutions for the 10 open questions raised in
[plans/2026-04-24-lift-lower-bivector-refactor.md](../plans/2026-04-24-lift-lower-bivector-refactor.md);
each is marked DECIDED (commit the resolution as written) or
NEEDS-USER-CALL (substantive ambiguity that should be confirmed
before implementation).  All proposed resolutions can be overridden;
treat this as a starting commitment, not a finished decision.

## Cross-references

- Plan: [plans/2026-04-24-lift-lower-bivector-refactor.md](../plans/2026-04-24-lift-lower-bivector-refactor.md)
- Conceptual ground truth: [Logic.md](../Logic.md) §8
- Ops inventory / mapping / divergences: [Logic.md](../Logic.md) §9 (formerly the standalone OpsComparison report).
- Mereology: [Mereology.md](../Mereology.md)
- Tetralemma semantics: [BuddhistParallels.md](../BuddhistParallels.md)
- Grammar: [../../data/grammar.cfg](../../data/grammar.cfg)
- Grammar in-progress rewrite: [../../todo.md](../../todo.md)

---

## O1 — Hard-pair extraction timing

**Question.**  Sparsity + cardinality + anneal on `SigmaLayer.weight`
and `PiLayer.weight` to produce `(op, c_a, c_b)` triples per symbol
— train-time annealing or post-training extraction pass?

**Resolution.**  DECIDED — **annealed during training**, with
inference-time hard-top-2 extraction available at any checkpoint.

**Schedule.**

```
λ_sparsity(epoch) = 0                                if epoch <  warmup_epochs
                  = λ_final * (epoch - warmup) / ramp  if warmup ≤ epoch < warmup + ramp
                  = λ_final                          otherwise

target_k = 2     # cardinality target (number of non-zero pooling weights per row)
```

**Defaults.**
- `warmup_epochs = total_epochs * 0.25`
- `ramp = total_epochs * 0.25` (so half the training is at full
  pressure)
- `λ_final = 0.1`
- Cardinality enforcement: straight-through-estimator on a top-k mask
  with `k = target_k`.  At inference, mask is hard top-2.

**Inference-time extraction.**

```python
class SigmaLayer:
    def extract_pair(self, symbol_idx: int) -> tuple[str, int, int]:
        """Return (op, c_a, c_b) for the named symbol.

        op = 'OR' for SigmaLayer extraction, 'AND' for PiLayer.
        c_a, c_b = top-2 indices of the pooling row.
        """
        row = self.weight[symbol_idx]
        top2 = row.abs().topk(2).indices
        return ('OR', int(top2[0]), int(top2[1]))
```

**Ablation hooks.**  Two independent config flags:
`sparsityEnabled`, `cardinalityEnabled`.  Allow turning either off
in isolation for ablation runs.  Both default `True` in production
configs.

**Why now.**  Defers no decisions; gives the codebook enough warmup
to develop content semantics before forcing interpretability;
extracted parses are usable without a separate post-training pass.

---

## O2 — Sigma readout: max vs averaging

**Question.**  Does the synthesis-as-union semantics use
softmax-weighted averaging, hard max, or arithmetic mean?

**Resolution.**  DECIDED — **softmax during training, hardmax at
inference**, with arithmetic mean kept as a deprecated `kind`
parameter for backward compat.

**API.**

```python
@staticmethod
def lift(X1, X2=None, mode='OR', kind='softmax', inverse=False,
         softmax_temperature=1.0):
    """Synthesis: many → one (∨)."""
    ...
```

`kind` values:
- `'softmax'` (default training): softmax-weighted average with
  `softmax_temperature`.  Smooth gradient.
- `'hardmax'` (inference / extraction): elementwise max.  Sub-
  differentiable, sparse gradient.
- `'mean'` (deprecated): arithmetic mean.  Carry-over from the old
  `Ops.lower = mean` body — kept under the Step 1 deprecation alias
  for one release.

**Temperature schedule.**  Optional anneal during training:

```
softmax_temperature(epoch) = 1.0 / (1.0 + anneal_rate * epoch)
```

with `anneal_rate = 0.0` by default (no anneal).  Setting
`anneal_rate > 0` gradually sharpens softmax toward hardmax over
training, complementing the O1 sparsity schedule.

**Why softmax default.**  Smoothing keeps gradient flowing through
all pooling weights early, so the codebook can settle before O1's
sparsity pressure forces the hard top-2.  Hardmax at inference
gives the categorical readout the "saturating
commitment" framing calls for.

---

## O3 — PerceptualSpace SigmaLayer

**Question.**  Does PerceptualSpace own a SigmaLayer for synthesis
from a sub-perceptual layer?

**Resolution.**  DECIDED — **defer**.

PerceptualSpace currently sits at the bottom of the stack.  It
produces percepts from raw input via the BPE chunking pipeline (see
[plans/2026-04-23-perceptualspace-bpe-chunking.md](../plans/2026-04-23-perceptualspace-bpe-chunking.md))
and has no layer below to synthesize from.  Owns `PiLayer` only
(analysis down toward sub-perceptual / byte features handled by the
chunker).

**Trigger to revisit.**  When sub-PerceptualSpace structure
(sub-BPE units, character-level features, learned subword
prototypes) becomes its own layer — at that point PerceptualSpace
gains a SigmaLayer for synthesis from below, and the sub-perceptual
layer gets a PiLayer for analysis to bytes.

---

## O4 — tetralemma_balance_penalty config sharing

**Question.**  Independent config for ConceptualSpace and
SymbolicSpace tetralemma policies, or shared?

**Resolution.**  DECIDED — **shared default with per-space
override**.

**Config schema.**  Add to the XML config:

```xml
<TetralemmaPolicy>
    <allowExcludedMiddle>1</allowExcludedMiddle>    <!-- permit NEITHER -->
    <allowContradiction>0</allowContradiction>     <!-- penalize BOTH -->
    <neitherThreshold>0.1</neitherThreshold>
</TetralemmaPolicy>

<ConceptualSpace>
    <!-- inherits TetralemmaPolicy by default -->
    <tetralemmaOverride enabled="false" />
</ConceptualSpace>

<SymbolicSpace>
    <tetralemmaOverride enabled="false" />
</SymbolicSpace>
```

When `tetralemmaOverride` is enabled, the per-space block is allowed
to define its own `<TetralemmaPolicy>`; otherwise the global policy
applies.

**Why shared.**  The two layers share semantics (same bivector
corners with the same Buddhist-parallels interpretation); divergence
in contradiction-tolerance would itself be a substantive design
choice that warrants explicit override.

---

## O5 — query and swap as new ops

**Question.**  Are `query` and `swap` first-class new ops, or
compositions of existing ops?

**Resolution.**  DECIDED — **both are new ops**, added under the
unified signature.

### Ops.query

**Signature.**

```python
@staticmethod
def query(X1, X2=None, mode='QUERY', inverse=False):
    """Interrogative speech act.

    Produces a state structurally similar to equals(X1, X2) but with:
      - Truth-axis dimensions zeroed (no commitment).
      - A 'query' marker on the where-axis (or a dedicated bit).
      - Composition of the NP/AP/NP arguments preserved for
        downstream answering.

    `inverse=True` returns the would-be answer state if the operands
    were committed (mostly used as a structural read).
    """
```

**Body sketch (monotonic mode).**

```python
def query(X1, X2=None, mode='QUERY', inverse=False):
    if X2 is None:
        # unary query: "what is X1?"
        truth_zero = torch.zeros_like(X1[..., :2])
        rest = X1[..., 2:]
        marked = mark_query_axis(rest)   # set query bit
        return torch.cat([truth_zero, marked], dim=-1)
    # binary: "is X1 X2?" / "is X1 part of X2?"
    proposition = Ops.equal(X1, X2)
    truth_zero = torch.zeros_like(proposition[..., :2])
    rest = proposition[..., 2:]
    marked = mark_query_axis(rest)
    return torch.cat([truth_zero, marked], dim=-1)
```

The `mark_query_axis` helper sets a dedicated bit (or low-magnitude
positional value) in the where slot so downstream code can
distinguish a question from a zeroed assertion.  Exact mechanism
depends on whether the where-encoding has a free dim — to be
finalized at implementation time.

### Ops.swap

**Signature.**

```python
@staticmethod
def swap(X1, X2=None, mode='SWAP', inverse=False):
    """Argument-position permutation.

    Binary form: returns a state whose internal argument-role
    structure is the swap of the input — for a relational state
    holding (subject, object), the result holds (object, subject).
    Useful for converting part(A, B) into whole(A, B) compositionally
    via swap(part(A, B)).

    Self-inverse: swap(swap(X)) == X.
    """
```

**Body.**  For activations whose role structure is encoded via the
bivector layout `[subj_pole, obj_pole, ...]`, `swap` is a positional
permutation on the role-marked dims.  For richer encodings, `swap`
applies the role-permutation matrix appropriate to the input's
relational type.

**Self-inverse property.**  Since position swap is involutive,
`swapReverse = swap`.

---

## O6 — bind vs intersection for asymmetric head-complement

**Question.**  Use `bind` as a separate op for asymmetric
head-complement structures (PP = P + NP, VP = V + NP), or route
through `intersection` with category-vector head-marking?

**Resolution.**  DECIDED — **Phase 1: route through `intersection`
with category-vector head-marking**; Phase 2: add `Ops.bind` only if
the parser needs first-class head extraction.

**Phase 1 mechanism.**

```python
def apply_rule_pp(p_state, np_state):
    pp_act = Ops.lower(p_state.activation, np_state.activation,
                       mode='AND')
    pp_state = State(category='PP')
    pp_state.activation = pp_act
    pp_state.head_marker = p_state.category_vector  # tag P as head
    return pp_state
```

The result is an intersection with a head-marker side-channel.
Downstream code that needs the head can read `state.head_marker`;
code that doesn't, treats the state as a normal intersection.

**Phase 2 trigger.**  When the grammar / parser needs `head(state)`
as a first-class queryable operation (e.g. for grammatical
agreement, head-driven parsing, semantic role labelling), promote
`bind` to a real op:

```python
@staticmethod
def bind(head, complement, inverse=False):
    """Asymmetric head-complement composition.
    Returns a state structurally distinguishing head from complement.
    """
```

Until that need arises, the side-channel is enough.

---

## O7 / O9 — scale for degree modification

**Question.**  How to model degree intensification (`AP = scale(DEG,
AP)`) — strict meet, multiplicative scale, or new op?

**Resolution.**  DECIDED — **new `Ops.scale` op with multiplicative
saturating semantics**.

**Signature.**

```python
@staticmethod
def scale(X_intensifier, X_target, inverse=False):
    """Multiplicative envelope intensification.

    Monotonic: scaled = X_target * (1 + |X_intensifier|), clamped.
    Bitonic:  scaled = X_target * (1 + |X_intensifier|), preserving sign.

    Linguistically: 'very red' is more-red along the red axis,
    not the meet of very-ness with red.  scale biases the target's
    saturation; strict meet would deflate red whenever DEG-magnitude
    is < 1.
    """
```

**Body.**

```python
def scale(X_intensifier, X_target, inverse=False):
    factor = 1.0 + X_intensifier.abs()  # ≥ 1
    if not inverse:
        scaled = X_target * factor
        return torch.clamp(scaled, -1.0, 1.0)
    # inverse: solve for X_intensifier given X_scaled and X_target
    # |intensifier| ≈ (|scaled| / |target|) - 1
    ratio = X_target.abs().clamp(min=epsilon)
    return (X_intensifier.abs() / ratio - 1.0).clamp(min=0.0)
```

**Why this shape.**  Saturating multiplication preserves the
linguistic property that intensifiers don't add new dimensions —
"very red" is more-strongly-red, not "very" ∧ "red" as separate
properties.  Strict meet (`AND/lower`) would un-saturate the AP
whenever DEG carries low magnitude, predicting that "slightly red" ⊆
"red" along the red axis with un-changed envelope, which is wrong.

**Reverse.**  Lossy without both `X_target` and `X_scaled` known.
The arity-2 reverse `(X_int, X_target) = scaleReverse(X_scaled)` is
under-determined in general; mark `reverse_exact = False`.

---

## O8 — NP = conjunction(NP, NP) semantics

**Question.**  Does AND-coordinated NP coordination produce the
intersection of the two NP regions (lattice meet) or the union
(lattice join)?

**Resolution.**  DECIDED — **propositional reading: lattice
intersection (AND/lower)**, per the grammar's explicit
`conjunction(NP, NP)`.

**Documented semantics.**

> When parsing *NP and NP* (e.g. "apples and oranges"), the
> resulting NP region is the lattice meet of the two conjunct
> regions.  Lattice-wise this is the intersection — for disjoint
> NPs ("apples" and "oranges") this is empty.  The linguistic
> "apples and oranges refers to a set containing both" effect is
> recovered at predicate consumption: when the coordinated NP is
> the argument of a predicate, the predicate distributes over both
> conjuncts via the `conjunction` operator's compositional
> structure.  For uses where the union is genuinely intended
> ("apples or oranges, whichever you have"), use the
> `disjunction(NP, NP)` rule (OR/lift, region union as convex
> hull).

**Test obligation.**  Verify on real corpus sentences during Step 6
implementation:
- "I like apples and oranges" — predicate "like" must distribute.
- "Apples and oranges are fruit" — predicate "are fruit" must hold
  of each conjunct (and trivially of the meet, if non-empty).
- "He picked apples or oranges" — single-conjunct semantics.

If the corpus consistently violates the propositional reading, fall
back to OR/lift for `conjunction(NP, NP)` and update the grammar.

**Why propositional.**  Two reasons:
1. The grammar in `todo.md` explicitly chose `conjunction` (not
   `disjunction`) for AND-coordination.  Honoring the rewrite
   keeps the lift/lower polarity coherent with the explicit-op
   form.
2. The convexity asymmetry argument (Logic.md §8): meet is exact,
   join is over-approximating.  Routing AND-coordination through
   the exact operation preserves more structure for downstream
   reasoning.

---

## O9 — DEG modification

**Question.**  Same as O7.

**Resolution.**  See O7.  `Ops.scale` covers it.

---

## O10 — Layer 2.5 reverse productions

**Question.**  Where do reverse productions live?

**Resolution.**  DECIDED — **derived mechanically at grammar-load
time** from the forward productions.

```python
def derive_reverse_rules(forward_rules):
    reverse_rules = []
    for rule in forward_rules:
        rev = Production(
            lhs=rule.args,                    # tuple
            op=rule.op + 'Reverse',
            args=[rule.lhs],                  # single
            reverse_exact=rule.reverse_exact, # carry the flag
        )
        reverse_rules.append(rev)
    return reverse_rules
```

The `<upward>` XML in `todo.md` is read as a **checklist** of
forward ops with working reverses, not a separate authored grammar
section.  Implementation reads it at load time only to verify that
every forward op has a corresponding `Ops.<op>Reverse` method
defined; missing reverses raise a grammar-load error with a
pointer to the missing implementation.

---

## Summary of new `Ops` methods to add

Net new methods introduced by these resolutions (beyond the existing
`Ops.lift`/`Ops.lower` rename + mode dispatch from Step 1):

| Method | Arity | Mode key | Reverse | Notes |
|---|---|---|---|---|
| `Ops.query` | 1 or 2 | `QUERY` | `queryReverse` (lossy, structural) | Interrogative speech act; truth-axis zeroed, query bit set |
| `Ops.swap` | 1 (binary-relation activation) | `SWAP` | `swapReverse = swap` (self-inverse) | Argument-position permutation |
| `Ops.scale` | 2 | `SCALE` | `scaleReverse` (lossy, under-determined) | Multiplicative saturating intensification |
| `Ops.project` | 1 | `PROJECT` | `projectReverse` (exact identity-strip) | Terminal projection; type-stamp LHS category |
| `Ops.parts` | 1 (+ codebook W) | n/a (query) | n/a — no inverse | All codebook rows that are parts of x |
| `Ops.wholes` | 1 (+ codebook W) | n/a (query) | n/a — no inverse | All codebook rows that have x as a part |

Optional / Phase 2:

| Method | Status |
|---|---|
| `Ops.bind` | Deferred — Phase 2 only if first-class head extraction is needed |

Existing methods that gain mode dispatch under the unified
signature: `lift`, `lower` (Step 1).  Existing methods that route to
modes but keep their current bodies: `conjunction`, `disjunction`,
`negation`, `non`, `part`, `whole`, `equal`, `overlap`, `underlap`,
`boundary`, `copart`, `what`, `where`, `when`, `distance`,
`negationReverse`, `conjunctionReverse`, `disjunctionReverse`.

---

## Summary of grammar.cfg commits

The unified table in
[plan Step 6](../plans/2026-04-24-lift-lower-bivector-refactor.md#step-6-grammar-driven-dispatch-via-datagrammarcfg)
captures every rule.  Substantive commits these resolutions
encode:

1. NP coordination uses the propositional reading
   (`conjunction = AND/lower`) per O8.
2. Degree modification uses `Ops.scale` per O7.
3. Head-complement uses `intersection` with head-marker tagging
   (Phase 1) per O6.
4. `query` and `swap` are new first-class ops per O5.
5. The `#`-marked uncertain rules in `todo.md` are resolved with
   `intersection(...)` defaults (`#S = MP S` → `intersection(MP,
   S)`, etc.); flagged ⚠ in the table for empirical confirmation
   during Step 6 implementation.
6. Reverse productions are derived at load time per O10.

---

## Configuration surface

New XML knobs introduced by these resolutions:

```xml
<HardPairExtraction>
    <sparsityEnabled>true</sparsityEnabled>
    <cardinalityEnabled>true</cardinalityEnabled>
    <warmupEpochs>0.25</warmupEpochs>          <!-- fraction of total -->
    <rampEpochs>0.25</rampEpochs>
    <lambdaFinal>0.1</lambdaFinal>
    <targetK>2</targetK>
</HardPairExtraction>

<SigmaReadout>
    <kindTraining>softmax</kindTraining>
    <kindInference>hardmax</kindInference>
    <softmaxTemperature>1.0</softmaxTemperature>
    <annealRate>0.0</annealRate>
</SigmaReadout>

<TetralemmaPolicy>
    <allowExcludedMiddle>1</allowExcludedMiddle>
    <allowContradiction>0</allowContradiction>
    <neitherThreshold>0.1</neitherThreshold>
</TetralemmaPolicy>

<ConceptualSpace>
    <tetralemmaOverride enabled="false" />
    <!-- bivector activation (Step 3) -->
    <bivectorActivation enabled="true" />
</ConceptualSpace>

<SymbolicSpace>
    <tetralemmaOverride enabled="false" />
</SymbolicSpace>
```

All values shown are the proposed defaults.

---

## Implementation milestones

Mapped onto the plan's Steps 1–7:

| Plan step | What lands | Spec resolutions touched |
|---|---|---|
| Step 1 (Ops rename + mode dispatch) | Unified signature, deprecation aliases | O2 (kind parameter scaffolding) |
| Step 2 (conjunction/disjunction forwarders) | Existing primitives route to new dispatcher | (none) |
| Step 3 (Bivector ConceptualSpace) | `[N, 2]` activation, signed-collapse shim | O4 (tetralemma config) |
| Step 4 (Pi/Sigma ownership flip) | Layers move to their owner spaces | (none) |
| Step 5 (Layer wiring) | PiLayer / SigmaLayer call `Ops.lower` / `Ops.lift` | O1 (sparsity hooks), O2 (softmax kind) |
| Step 6 (Grammar dispatch) | Loader, dispatcher, category vectors, reverse derivation, soft-superposition path | O5 (query/swap), O6 (bind/intersection), O7/O9 (scale), O8 (NP coord), O10 (reverse), Q3 (nested-call ternary), Q4 (VO category vector), Q5 (deprecation aliases stay), **R1 (rule-level soft superposition)** |
| Step 7 (Shim removal) | Bivector consumers migrated | (none) |

Steps 1, 3, 4, 5 are independently revertible per the plan.  Step 6
gates on Step 1.

---

## Parts and wholes — set-returning mereological queries (P1–P2)

The pointwise `Ops.part(x, y, scalar=True)` and `Ops.whole(x, y,
scalar=True)` answer the question "is x a part of y?" for one
specific pair.  Reasoning frequently needs the **set** of all parts
of a given symbol, or the set of all wholes that contain it.  Two
new methods make those queries first-class:

```
parts(x, W=None, threshold=0.7, monotonic=True)  -> indices, scores
wholes(x, W=None, threshold=0.7, monotonic=True) -> indices, scores
```

These are symbol-level (Layer 2) operations: they consume an
existing activation `x` plus a codebook `W` (or a stored prototype
set), walk every row, and return the rows that satisfy the
mereological threshold.

### P1 — `Ops.parts(x, W, threshold=0.7, monotonic=True)`

DECIDED.  Returns a **dense codebook-shaped score vector** in which
every entry is either `0` (below threshold — masked out) or the
above-threshold part-score.  Shape `(K,)` matches the codebook
dimensionality, so the result acts as both a *mask* (zero entries
indicate non-parts) and a *significance vector* (non-zero entries
allow ranked selection or further weighting).

**Signature.**

```python
@staticmethod
def parts(x, W=None, threshold=0.7, monotonic=True) -> torch.Tensor:
    """Dense per-codebook-row part-scores of x.

    Parameters
    ----------
    x : Tensor of shape (..., D)
        Query symbol activation.  Last dim is the bivector .what
        (or the high-dim concept embedding for ConceptualSpace).
    W : Tensor of shape (K, D), optional
        Codebook of K candidate prototypes.  If None, the caller
        must wrap via Basis.parts which supplies self.getW().
    threshold : float, default 0.7
        Minimum scalar parthood score; entries strictly below this
        are zeroed.
    monotonic : bool, default True
        Bivector mode for bivector layers; bitonic supported but
        uncommon at this layer.

    Returns
    -------
    scores : FloatTensor of shape (K,)
        Per-codebook-row part-scores in [0, 1].  Entries below
        threshold are zero (mask); entries at or above threshold
        carry their score (significance).  Length equals the
        codebook size, so the result can be broadcast / multiplied
        against the codebook directly without index-select.
    """
```

**Body.**

```python
@staticmethod
def parts(x, W=None, threshold=0.7, monotonic=True):
    if W is None or W.shape[0] == 0:
        return torch.empty(0, dtype=x.dtype, device=x.device)

    K = W.shape[0]
    # Broadcast x against every row of W.
    x_expanded = x.unsqueeze(0).expand(K, *x.shape)   # (K, ..., D)
    W_expanded = W.view(K, *([1] * (x.ndim - 1)), W.shape[-1])
    W_expanded = W_expanded.expand(K, *x.shape[:-1], W.shape[-1])

    # Score: part(w_i, x, scalar=True)
    scores = Ops.part(W_expanded, x_expanded,
                      monotonic=monotonic, scalar=True)
    # Reduce leading non-K dims (batch, concept-position) to one
    # score per codebook row.
    flat_scores = scores.reshape(K, -1).mean(dim=-1)

    # Zero below-threshold entries — caller gets a mask + significance
    # vector in one tensor.
    return torch.where(
        flat_scores >= threshold,
        flat_scores,
        torch.zeros_like(flat_scores),
    )
```

**Caller idioms.**

```python
mask  = ws.symbolic.basis.parts(x)       # (K,) — zero or score
top_k = mask.topk(k=5).indices            # ranked indices
gated = mask.unsqueeze(-1) * codebook    # broadcast-weighted codebook
nonzero_idx = mask.nonzero(as_tuple=False).squeeze(-1)
```

The dense form lets the caller use the result as either a soft
attention mask (multiply) or a hard selector (`nonzero` /
`topk`).  Index-only callers pay one extra `nonzero` call; mask
callers get the result directly with no allocation.

### P2 — `Ops.wholes(x, W, threshold=0.7, monotonic=True)`

DECIDED.  Same dense `(K,)` return shape as `parts`; the only
difference is the part-call has its arguments swapped (the converse
relation).

**Signature.**  Identical to `parts`.

**Body.**

```python
@staticmethod
def wholes(x, W=None, threshold=0.7, monotonic=True):
    if W is None or W.shape[0] == 0:
        return torch.empty(0, dtype=x.dtype, device=x.device)

    K = W.shape[0]
    x_expanded = x.unsqueeze(0).expand(K, *x.shape)
    W_expanded = W.view(K, *([1] * (x.ndim - 1)), W.shape[-1])
    W_expanded = W_expanded.expand(K, *x.shape[:-1], W.shape[-1])

    # Score: part(x, w_i, scalar=True) — the converse
    scores = Ops.part(x_expanded, W_expanded,
                      monotonic=monotonic, scalar=True)
    flat_scores = scores.reshape(K, -1).mean(dim=-1)

    return torch.where(
        flat_scores >= threshold,
        flat_scores,
        torch.zeros_like(flat_scores),
    )
```

**Identity.**  By construction `wholes(x, W)` is `parts(x, W)`
with the part-call's arguments swapped — the same dual relationship
as `whole(x, y) = part(y, x)` at the pointwise level.  A single
internal helper implements both:

```python
def _mereological_query(x, W, threshold, monotonic, swap):
    ...
    if swap:
        scores = Ops.part(x_expanded, W_expanded, ...)   # wholes
    else:
        scores = Ops.part(W_expanded, x_expanded, ...)   # parts
    flat = scores.reshape(K, -1).mean(dim=-1)
    return torch.where(flat >= threshold, flat, torch.zeros_like(flat))
```

### Symmetry caveat under clipped cosine

Under the current clipped-cosine `part` ([Layers.py:4395](../../bin/Layers.py)),
`part(A, B) = part(B, A)` because cosine is symmetric and norms are
sign-invariant (see [Mereology.md](../Mereology.md) *Asymmetric
subsumption*).  Therefore **`parts(x, W)` and `wholes(x, W)`
return the same set under clipped cosine**.

Asymmetric subsumption is recovered relationally via figure /
ground:

```python
parts_strict(x, W, neg_W) = parts(x, W) ∩ wholes(neg_W, W')
```

where `neg_W` is the codebook with each row negated and `W'` the
relevant subset.  This is the "compare `part(A, B)` against `part(A,
¬B)`" recipe from Mereology.md, lifted to set-returning form.

For a future `Basis` with truly asymmetric `part` (e.g. a
codebook-search-based parthood that distinguishes ⊆ from ⊇), the
two methods diverge naturally and no recipe-via-negation is needed.

### `Basis` wrappers

Each of the three Spaces gains `Basis.parts(x, threshold=0.7)` and
`Basis.wholes(x, threshold=0.7)` thin wrappers that supply
`W = self.getW()` and forward the layer's `monotonic` flag:

```python
def parts(self, x, threshold=0.7):
    return Ops.parts(x, W=self.getW(),
                     threshold=threshold,
                     monotonic=self.monotonic)

def wholes(self, x, threshold=0.7):
    return Ops.wholes(x, W=self.getW(),
                      threshold=threshold,
                      monotonic=self.monotonic)
```

This matches the existing `Basis.conjunctionReverse` /
`Basis.disjunctionReverse` pattern at [Spaces.py:936–958](../../bin/Spaces.py).

### Use sites

**Reasoning / inference.**  `TruthLayer.derive` at
[Layers.py:2636](../../bin/Layers.py) currently performs pairwise
mereological inference via the Grammar's `part` rule with a hard-
coded loop.  After P1 / P2 land, `derive` can call
`wordSpace.symbolicSpace.basis.wholes(stored_truth_i)` to find
candidate "X-of-which-this-is-a-part" entries in one batched call,
replacing the manual inner loop.

**Grammar dispatch.**  When the grammar's `S = part(NP, NP)`
production fires, the dispatcher can use `parts` / `wholes` to
verify the relation against the stored truth set or the codebook,
returning evidence for / against the assertion.

**ImpenetrableLayer.**  The full `K × K` parthood matrix
([Layers.py:2035](../../bin/Layers.py)) computes, in effect, `parts(cb_i,
cb)` for every `i`.  P1 expresses the per-row case as a public op;
the layer's matrix view remains the regularizer's internal form.

**Truth queries.**  When the user asks "what are the parts of X?"
or "what is X a part of?", the answer is `parts(X)` / `wholes(X)`
applied to the symbolic codebook (or the stored truth set,
depending on whether the question concerns the type system or the
extension).

### Threshold defaults and configuration

Default `threshold = 0.7` matches `TruthLayer.derive`'s default
([Layers.py:2636](../../bin/Layers.py)) and the
`fullPartThreshold = 0.9` / `disjointThreshold = 0.1` framing in
[Mereology.md](../Mereology.md) *ImpenetrableLayer Configuration*.
The mereology suite's "full part" threshold τ ≈ 0.9 is stricter
than the inference threshold; `parts` / `wholes` use the looser
inference threshold by default so they surface candidate parts
even when the relation is partial.  Strict-only queries pass
`threshold=0.9`.

XML knobs:

```xml
<MereologicalQuery>
    <inferenceThreshold>0.7</inferenceThreshold>   <!-- parts() / wholes() default -->
    <strictThreshold>0.9</strictThreshold>          <!-- full part / equal -->
</MereologicalQuery>
```

### Reverses

`parts` and `wholes` are **queries**, not transformations — they do
not have a meaningful inverse in the lift / lower sense.  No
`partsReverse` / `wholesReverse` is added.  Where the grammar's
inverse path needs to recover a witness pair, it uses
`Ops.partReverse` (which supplies the converse witness) or routes
through `conjunctionReverse` / `disjunctionReverse` for the
codebook-search recovery.

### Sub-row reduction in the leading-axis case

For `x` of shape `(D,)` (a single bivector — pre-batch, pre-
position), the implementation skips the reduction and returns
exact per-row scores.  For `x` of shape `(B, N, D)`, the default
mean-over-leading-axes is reasonable but loses per-position
structure; callers needing per-position results either iterate or
flatten upstream.  An optional `reduce='mean'|'max'|'none'` keyword
can be added if the per-position case becomes load-bearing — flag
to revisit during implementation.

---

## Bivector representation across model layers (B1–B7)

The discussions converged on representing per-concept activations as
**bivector pairs `[aP, aN]`** — positive and negative evidence on
independent dimensions — rather than signed scalars.  The four
corners of the pair encode the tetralemma:

```
[1, 0] = TRUE       [0, 0] = NEITHER (no commitment)
[0, 1] = FALSE      [1, 1] = BOTH    (contradiction)
```

This subsumes O4 above and adds explicit decisions B1–B7 below for
the model files that consume or produce activations.

### B1 — `ConceptualSpace.subspace.activation` shape: `[N, 2]`

DECIDED.  The conceptual activation tensor changes from a single
signed scalar per concept to a `(pos, neg)` pair per concept, in the
unsigned `[0, 1]` (monotonic) layout.

**Files.** [`bin/Spaces.py`](../../bin/Spaces.py) — `ConceptualSpace.__init__`
mirrors what SymbolicSpace already does at line 5933:

```python
class ConceptualSpace(Space):
    def __init__(self, inputShape, spaceShape, outputShape, ...):
        super().__init__(...)
        # Concepts carry 4-valued (quaternary) truth in .what via a 2-dim
        # bivector [pos_pole, neg_pole].  Mirrors SymbolicSpace.
        self.subspace.nWhat = 2
        self.subspace.muxedSize = 2 + self.subspace.nWhere + self.subspace.nWhen
        ...
```

**Backward-compatibility shim.**  `ConceptualSpace.activation_signed()`
returns `aP − aN` (the legacy single-scalar form) for callers that
have not migrated.  Removed in plan Step 7.

**What changes.**
- `cs.subspace.activation.shape == (B, N, 2 + nWhere + nWhen)` (was
  `(B, N, contentDim + nWhere + nWhen)` with a wider content slot).
- The codebook of conceptual prototypes also drops to `nWhat = 2`
  unless a separate "rich-content" codebook is needed.  See B4.
- All `Ops` calls on conceptual activations consistently set
  `monotonic=True` so they read the bivector as paired-index, not
  signed.

**Why `monotonic=True` (not bitonic) at the conceptual layer.**
Bivector `[aP, aN]` is the canonical unsigned representation;
bitonic `[-1, 1]` collapses both poles onto a single dimension and
loses the contradiction / ignorance distinction the Buddhist-
parallels framing depends on.  The whole point of the refactor is
to keep `[1, 1]` and `[0, 0]` apart at the conceptual layer.

### B2 — Lift / lower preserve the bivector

DECIDED.  Both directions of the level-crossing pair operate on
`[N, 2]` activations end-to-end:

- **Lift (C → S).**  Each concept's `(aP, aN)` flows into the
  symbol via Sigma's pooled synthesis without collapsing to a
  scalar at any intermediate step.  The synthesized symbol's
  `.what` is itself `[2K]` paired-index (bivector across K pooled
  concepts) — the existing SymbolicSpace layout.
- **Lower (S → C).**  Each pooled symbol decomposes via Pi back
  into per-concept `(aP, aN)` pairs.  Pi's weighted-product factor
  recovery preserves both poles per concept; no signed-sum
  collapse.

**Forbidden collapse.**  The legacy "signed-sum to a single scalar
per concept" is removed.  Where downstream code needs a scalar:

```python
signed        = aP - aN          # canonical truth direction
contradiction = aP * aN          # high where both poles fire
ignorance     = (1 - aP) * (1 - aN)  # high where neither pole fires
```

These three derivable scalars are computed at the consumer, not
baked into the activation upstream.

### B3 — `PerceptualSpace` adopts bivector activation

DECIDED — for consistency.  PerceptualSpace's activation also
becomes `[N, 2]` bivector (`monotonic=True`).  Without this,
percept-to-concept lift inherits a scalar-to-bivector promotion
that has to be manufactured at the C boundary, which is awkward.

Concretely, the percept-bivector is an attention-weighted
`(observed_intensity, expected_absence)` pair.  Default mapping
from current scalar percept activations:

```python
aP = max(0, x)
aN = max(0, -x)   # if percepts were bitonic; else 0
```

Sub-perceptual layers (BPE chunking et al.) feed into this with
their own conventions; PerceptualSpace promotes to bivector at
ingestion.

**Trade-off.**  Doubles the percept activation storage.  Acceptable
because percept N is small relative to concept N and the consistency
gain across the lift / lower chain is substantial.

### B4 — Codebook `.what` widths: high-dim conceptual, bivector symbolic

DECIDED — **codebook `.what` differs by layer**.

| Codebook | `.what` width | Activation `.what` width | Notes |
|---|---|---|---|
| PerceptualSpace | bivector (`nWhat = 2`) | bivector (`nWhat = 2`) | per B3 |
| ConceptualSpace | **high-dim** (`nWhat = D_concept`) | **bivector (`nWhat = 2`)** | concept prototypes carry rich content; activations summarize per-concept presence |
| SymbolicSpace | bivector (`nWhat = 2`) | bivector (`nWhat = 2`) | unchanged from current |
| WordSpace category codebook (`category_codebook`) | bivector (`nWhat = 2 + nWhere + nWhen`) | n/a | bivector for graded soft-superposition commitment, but separate from content; see B6 |

**The asymmetry that matters at ConceptualSpace.**  Concept
prototypes need rich content to encode "apple-ness" or "redness"
— two dimensions are not enough.  But the per-timestep *activation*
against that codebook is the (positive, negative) match score for
each concept — a bivector per concept, regardless of how wide the
prototype is.  So for ConceptualSpace:

- `subspace.activation.shape == (B, N_concepts, 2 + nWhere + nWhen)` (bivector)
- `subspace.basis.getW().shape == (K_concepts, D_concept + nWhere + nWhen)` (high-dim)

Operations that consume *activations* use `monotonic=True` on the
bivector; operations that consume *prototypes directly* (the
codebook in `ImpenetrableLayer._pairwise_parthood`, for example)
use whatever metric is appropriate for the high-dim space (clipped
cosine `part` is invariant to `nWhat` width).

The Pi / Sigma layers are precisely what bridge the two widths:
**lower / analysis** (S → C) projects from a bivector symbol back
into a high-dim concept envelope and emits a bivector activation
per concept; **lift / synthesis** (C → S) gathers high-dim concept
content into a bivector symbol.  The shape change at the C / S
boundary is intrinsic to the architecture, not a bug.

PerceptualSpace and SymbolicSpace keep prototype and activation
`.what` aligned (both bivector).  ConceptualSpace is the only space
where the two diverge, by design.

**Open: `D_concept`.**  Default width for the conceptual codebook
prototypes is whatever the existing config sets — the refactor does
not change it.  If a concrete value is needed for the spec,
`D_concept = 32` is a reasonable starting placeholder; revisit at
implementation time per measured embedding capacity.

### B5 — `monotonic=True` propagates through every Basis call on bivector layers

DECIDED.  The `monotonic` flag on `Ops` and `Basis` methods selects
between bitonic `[-1, 1]` and monotonic `[0, 1]` paired-index
arithmetic.  Once ConceptualSpace and PerceptualSpace are
bivector, every call site that operates on their activations must
pass `monotonic=True` (matching what SymbolicSpace already does).

**Inventory of call sites that need the propagation.**

| Site | Current state | After |
|---|---|---|
| `Basis.conjunction / disjunction / negation / non / part / whole / equal / overlap / underlap / boundary / copart / distance` ([Spaces.py:910–998](../../bin/Spaces.py)) | Already accept `monotonic` parameter | No change in signature; callers must pass `monotonic=True` for conceptual/perceptual layers |
| `SyntacticLayer._mono(subspace)` ([Language.py:710-ish](../../bin/Language.py)) | Returns `subspace.monotonic` flag | Confirm flag is `True` for ConceptualSpace's subspace after B1 lands |
| `Ops.distance` ([Layers.py:4370](../../bin/Layers.py)) | Bitonic = angular, monotonic = volume-weighted L2 | Conceptual distance routes to volume-weighted L2 |
| `Ops.part / equal / overlap / underlap / boundary / copart` ([Layers.py:4395–4495](../../bin/Layers.py)) | `monotonic` and `scalar` parameters | Conceptual `part(scalar=True)` must pass `monotonic=True` so the empty-operand contract reads bivector zeros correctly |
| `TruthLayer.luminosity / fusion` ([Layers.py:2400–2549](../../bin/Layers.py)) | Already bivector-aware (uses paired-index `_positive_poles` / `_negative_poles`) | Confirm slicing matches the **leading-bivector** layout of bivector ConceptualSpace — see B7 below |
| `ImpenetrableLayer._pairwise_parthood` ([Layers.py:2035](../../bin/Layers.py)) | Uses `Basis.part` on codebook rows | Conceptual codebook regularization must pass `monotonic=True` |
| `Loss` classes (`ModelLoss`, `CertaintyWeightedMAELoss`, `CertaintyWeightedMSELoss`, `CertaintyWeightedCrossEntropy`) ([Layers.py:4499+](../../bin/Layers.py)) | Consume scalar activations | After B1: consume bivector and either (a) call `signed = aP - aN` internally and continue, or (b) penalize directly on the bivector with sign-flip semantics |

**Decision rule.**  Any code path that previously operated on a
ConceptualSpace activation as a signed scalar:

- If it just needs the truth direction → use `aP - aN` (signed
  shim) and call existing logic.
- If it needs to distinguish contradiction from ignorance → consume
  the bivector directly and branch on `aP * aN` and `(1-aP)*(1-aN)`.
- If it does composition / lift / lower / part / equal → pass
  `monotonic=True` and let the bivector primitives handle it.

**Explicit `monotonic` defaults at construction time.**  Each
Space's `Basis` carries a `monotonic` flag set at construction
([Spaces.py:770, 787](../../bin/Spaces.py)).  After B1 / B3 land:

```python
PerceptualSpace.subspace.monotonic = True
ConceptualSpace.subspace.monotonic = True
SymbolicSpace.subspace.monotonic   = True   # already
WordSpace.subspace.monotonic       = True   # follows from upstream
```

The `_mono(subspace)` helper used by `SyntacticLayer` rule bodies
returns this flag, so once it is uniformly `True` across the
conceptual / symbolic stack, the bivector path is consistent
end-to-end.

### B6 — Category vectors are bivector (for soft-superposition loss flow), in their own codebook

DECIDED — categories carry **bivector** `(aP, aN)` per category
axis, **but stay in their own dedicated `category_codebook`** rather
than merging with content concepts.  Two reasons in tension; the
bivector form addresses one, separate-codebook addresses the other.

**Why bivector — soft superposition needs graded commitment.**
The model trains via soft superposition over rules: at each parse
step, multiple rules fire with weighted probabilities so the loss
can flow through the right combination.  This requires the
*categorical state* of every parser state to be representable as a
graded commitment, not a hard tag:

```
[1, 0]      = fully committed to category-X
[0, 1]      = fully committed to NOT-X
[0, 0]      = no commitment yet (free to bind)
[0.7, 0.4]  = mostly X, with some not-X evidence (mid-resolution)
[1, 1]      = contradictory (training pressure to resolve)
```

A 4-dim opaque categorical embedding cannot express graded
commitment with structural endpoints — "uncertain between ADJ and
N" has no canonical encoding, and the gradient signal during
training has nowhere to push toward.  The bivector pole-pair gives
the model a clean variable to *resolve*: training pressure drives
each state's category bivector toward saturation on the right axis,
making the eventual hard-inference readout (argmax of the active
poles) sharp and unambiguous.

**Efficiency consequence.**  Once the category is resolved (or
mostly resolved), the rule dispatcher prunes its candidate rule
set to those whose LHS / RHS slots match the saturated category.
A diffuse category bivector keeps many rules in superposition; a
saturated one focuses the loss signal through a single rule
application.  Without a graded category form, every rule fires at
every step in proportion to whatever ad-hoc similarity the
embedding produces — diffuse loss, slow convergence.

**Why a separate codebook — categories are types, not content.**
ADJ-ness is not a concept in the same sense that "apple" is.  It
has no truth value, no parts, no wholes — querying `parts(ADJ)`
returning all sub-categories of ADJ-ness is not a thing the parser
ever asks for.  Putting categories in the conceptual content
codebook would invite:

- Wasted `ImpenetrableLayer` regularizer pressure pushing ADJ vs N
  to `disjoint` — they already are by construction (enumerated
  tags); training shouldn't fight to discover what's already
  declared.
- Cross-contamination during VQ updates if a content concept's
  embedding drifts toward a category region.
- Architecture-diagram confusion (the same codebook holding both
  "apple" and "noun-ness" makes failure modes harder to debug).

**Result: bivector `category_codebook`, separate from content.**
Concrete layout:

```python
# WordSpace.__init__ (Language.py:2035-2050 area)
self.category_codebook = Codebook()
self.category_codebook.create(
    nVectors=K_categories,    # one row per (S, NP, VP, AP, MP, PP, DEF, HAS,
                              #              N, V, ADJ, ADV, IS, POSSESS, NOT,
                              #              AND, OR, P, DET, DEG, VO, ...)
    nDim=2 + nWhere + nWhen,  # bivector .what plus optional positional
    monotonic=True,           # bivector mode
)
```

**Each row** is initialized with `[1, 0]` on the row's own axis,
`[0, 1]` on the not-this-category axis, but in the simpler
single-axis-per-category model each row's bivector is just its
own commitment polarity:

```
category_codebook[idx_NP] = [1, 0, ...positional...]
```

**Each parser state** carries a category bivector that starts at
`[0, 0]` (no commitment) and migrates toward `[1, 0]` (committed)
through training as the rule predictor refines its rule choice.

**What this simplifies relative to merged-with-content (the
earlier framing).**

- Categories don't go through `ImpenetrableLayer`'s pairwise-
  parthood matrix.  They are mutually disjoint by construction.
- `parts()` / `wholes()` on category vectors return empty (or
  whatever the threshold yields against the small category
  codebook only) — they don't accidentally surface content
  concepts.
- The conceptual content codebook stays focused on content; its
  K rows aren't padded with K_categories category rows.

**What this simplifies relative to flat-tag-only (the previous
revision).**

- The training loss signal can flow through category commitment
  itself, not just through rule choice.  When the rule predictor
  is confused about whether a state is ADJ or N, the bivector
  carries that uncertainty into the next layer rather than
  collapsing it prematurely.
- `Ops.negation(monotonic=True)` on a category bivector swaps
  poles: turning ADJ into not-ADJ in one pole-flip.  Useful for
  negative agreement constraints ("must NOT be a verb").
- `Ops.non(monotonic=False)` (triangular residual) gives a free
  "uncommitted-ness" measure: high when neither pole has fired,
  low at saturation.  Useful as a per-state confidence readout.
- `tetralemma_balance_penalty` over category bivectors gives a
  natural training pressure: penalize `[1, 1]` (contradictory
  category commitment) as a regularizer that nudges the parser
  toward decisive type assignment.

**Migration plan.**  `WordSpace.category_codebook` already exists
and is currently 4-dim ([Language.py:2035](../../bin/Language.py)).
The refactor changes its `nDim` from 4 to `2 + nWhere + nWhen` and
sets `monotonic=True`.  Existing call sites that read category
embeddings (`category_lookup`, `pos_lookup`, the rule predictor)
adapt to the bivector shape with the same `monotonic=True` /
bivector-aware patterns documented in B5.

### B7 — Bivector layout: leading-bivector everywhere

DECIDED — **leading-bivector layout in every place that holds
bivectors**, including `TruthLayer`.  The paired-index layout
`[p0, n0, p1, n1, ...]` is retired.

**Layout commitment.**

```
[pos_pole, neg_pole, ...rest]
```

— a single 2-dim bivector at the leading positions of the last
axis, with whatever follows being the per-instance content
(positional trailers `where`, `when`, or — at the codebook level
in ConceptualSpace per B4 — the high-dim concept embedding tail).

**What changes vs the existing code.**

`TruthLayer.luminosity` ([Layers.py:2400+](../../bin/Layers.py)) and
`TruthLayer.fusion` ([Layers.py:2523+](../../bin/Layers.py)) currently
use paired-index slicing helpers `_positive_poles` (`[..., 0::2]`)
and `_negative_poles` (`[..., 1::2]`).  Under leading-bivector,
those become:

```python
def _positive_pole(self, v):
    """Leading positive pole — index 0 of the last axis."""
    return v[..., 0]

def _negative_pole(self, v):
    """Leading negative pole — index 1 of the last axis."""
    return v[..., 1]
```

(Singular, not plural — there is one leading bivector per row, not
K paired bivectors per row.)

`TruthLayer.tetralemma_balance_penalty` ([Layers.py:2464+](../../bin/Layers.py))
loses the layout-mismatch warning entirely.  Multi-concept
luminosity / fusion / contradiction is reduced over the *batch*
axis (across stored truths) rather than over an internal paired
axis — every truth contributes one bivector, and the elementwise
min / max / product across truths gives the conjunction / disjunction
/ contradiction reduction directly.

**Callers that previously sliced `[..., :2]`** can drop the slice
where the activation is already leading-bivector; the slice is now
a no-op.  Where the activation has a bivector at the *front* of a
larger row (true after the refactor in every space), the same
`acts[..., :2]` extraction is the canonical way to get the
bivector pair, and the rest of the row is the per-instance content
or positional trailers.

**Migration risk.**  TruthLayer's slicing change touches a load-
bearing reduction; the test surface for `luminosity`, `fusion`, and
`tetralemma_balance_penalty` should run pre/post the layout switch
and confirm bit-equivalence on stored-truth fixtures whose poles
are written in the new layout.

### B-summary — what each Space owns

After B1–B7:

| Space | `subspace.nWhat` | `monotonic` | activation shape | Owns layers |
|---|---|---|---|---|
| InputSpace | (unchanged — byte-buffer) | n/a | per token | (none in lift/lower family) |
| PerceptualSpace | 2 | True | `[B, N, 2 + nWhere + nWhen]` | PiLayer |
| ConceptualSpace | activation 2; codebook `D_concept` (high-dim per B4) | True | activation `[B, N, 2 + nWhere + nWhen]`; codebook `[K, D_concept + nWhere + nWhen]` | SigmaLayer (synthesis from P), PiLayer (analysis to S) |
| SymbolicSpace | 2 (already) | True (already) | `[B, N, 2 + nWhere + nWhen]` | SigmaLayer (synthesis from C) |
| WordSpace | (follows symbolic) | True | (carries category_stack, not lift/lower path) | TruthLayer, ImpenetrableLayer, InterSentenceLayer (existing) |
| OutputSpace | (downstream-defined) | depends on consumer | (downstream) | (downstream) |

---

## Updated configuration surface (with bivector knobs)

```xml
<TetralemmaPolicy>
    <allowExcludedMiddle>1</allowExcludedMiddle>
    <allowContradiction>0</allowContradiction>
    <neitherThreshold>0.1</neitherThreshold>
</TetralemmaPolicy>

<PerceptualSpace>
    <bivectorActivation enabled="true" />   <!-- B3 -->
    <tetralemmaOverride enabled="false" />
</PerceptualSpace>

<ConceptualSpace>
    <bivectorActivation enabled="true" />   <!-- B1 -->
    <tetralemmaOverride enabled="false" />
    <activationSignedShim enabled="true" />  <!-- removed in plan Step 7 -->
</ConceptualSpace>

<SymbolicSpace>
    <tetralemmaOverride enabled="false" />
</SymbolicSpace>
```

(Other knobs from the earlier *Configuration surface* section
remain unchanged.)

---

## Updated implementation milestones (B-row mapping)

| Plan step | Bivector / monotonic resolutions touched |
|---|---|
| Step 1 (Ops rename + mode dispatch) | B5 (audit `monotonic` propagation paths in dispatcher signatures) |
| Step 2 (conjunction/disjunction forwarders) | B5 (ensure forwarders accept and forward `monotonic` correctly) |
| Step 3 (Bivector ConceptualSpace) | **B1, B2, B5, B7** — primary landing point for bivector activation, signed shim, layout caveat |
| Step 4 (Pi/Sigma ownership flip) | B2 (lift / lower preserve bivector through the new ownership wiring) |
| Step 5 (Layer wiring) | B2, B5 (PiLayer / SigmaLayer route through `Ops.lift` / `Ops.lower` with `monotonic=True`) |
| Step 6 (Grammar dispatch) | **B6** — category vectors as bivector type-tags in `WordSpace.category_codebook` (separate from content); **R1** — rule-level soft superposition with both-branch propagation |
| Step 7 (Shim removal) | B1 — remove `activation_signed` shim once consumers are on bivector |

A new sub-step inside Step 3 covers **B3 (PerceptualSpace bivector
adoption)** — small enough to land alongside ConceptualSpace's
shape change rather than as a separate step.

---

## Rule-level soft superposition (R1)

### R1 — Both branches propagate; weights `w` and `1 − w` (or the R-ary softmax)

DECIDED.  At every rule-choice decision in the parser, **all
candidate rules fire**, each contributing its result to the LHS
state's activation, weighted by the rule predictor's transition
probability.  The binary case uses `(w, 1 − w)`; the R-ary case
uses the full softmax distribution `[p_1, …, p_R]` over candidate
rules.

**Why.**  Bivector categories (B6) act as a Straight-Through
Estimator with a contradiction sensor: they let the parser
*resolve* its category commitment over training, with the
tetralemma `aP * aN` pole-product giving a regularizer signal
when the commitment is internally inconsistent.  But this
mechanism only sees the path the model took.  It can correct
inconsistency *within* a chosen rule application but cannot
inform the model about the path it didn't take.  Without
both-branch propagation the optimizer never learns that rule B
would have been better than rule A — it only learns to clean up A.

The `1 − w` propagation closes this loop.  Each branch's result
contributes to the loss weighted by its rule probability;
gradient flows through every branch in proportion; the rule
predictor learns which branch was actually right.

**Forward path.**

```python
def apply_rules(rule_candidates, rhs_state_lookup, predictor_input):
    """Soft-superposition rule application.

    rule_candidates is the list of rules whose LHS matches the
    target category and whose RHS slots can be filled from the
    available state set.  predictor_input is whatever the rule
    predictor reads (typically the flattened category stack).
    """
    # Rule predictor outputs a softmax distribution over candidates.
    weights = softmax(predictor_input @ predictor_W, dim=-1)   # (R,)

    # Each candidate rule produces its result independently.
    results = []
    for rule, w in zip(rule_candidates, weights):
        rhs_states = rhs_state_lookup(rule)
        # Apply rule's op (Layer 1 dispatch) to its arguments.
        result = run_rule(rule, rhs_states)
        results.append(w * result)

    # Weighted sum is the LHS state's activation.
    lhs_act = sum(results)
    return lhs_act
```

**Inference path.**

```python
def apply_rules_hard(rule_candidates, rhs_state_lookup, predictor_input):
    """Argmax rule selection at inference time."""
    logits = predictor_input @ predictor_W   # (R,)
    chosen_idx = logits.argmax()
    chosen_rule = rule_candidates[chosen_idx]
    rhs_states = rhs_state_lookup(chosen_rule)
    return run_rule(chosen_rule, rhs_states)
```

**Bivector category interaction.**  The LHS state's category
bivector during soft superposition is the weighted mixture of
each branch's category prediction:

```python
lhs_category_bivector = sum(w * branch.lhs_category_bivector
                             for w, branch in zip(weights, branches))
```

If all candidate branches commit to the same LHS category (the
common case — every `NP -> …` rule produces an NP), the mixture
saturates on the NP axis regardless of `w`.  If branches commit
to *different* LHS categories (rare; mostly the closed-class
auxiliary productions), the mixture under-saturates and the
tetralemma regularizer applies pressure to resolve, exactly as
B6 designed.

**Annealing the softmax.**  Training starts with a soft softmax
(temperature `T = 1.0`) so every branch contributes meaningfully;
training anneals toward sharper softmax (`T → 0` or hardmax) as
the rule predictor learns.  Same pattern as O2's `softmax_temperature`
for SigmaLayer pooling.  The two anneals can share a single
schedule or run independently — flag for review at implementation
time.

**Compute cost.**  Soft superposition runs every candidate rule
at every decision point, so training compute scales with
*candidate-set size per rule choice*.  In practice the candidate
set is small (≤ 5 productions per LHS in `data/grammar.cfg`); the
overhead is bounded.  Inference is unaffected (single-branch
argmax).

**Relationship to existing soft-superposition machinery.**  The
parser already runs in soft-superposition mode during training
(see `_superposed_op` at [bin/Spaces.py:6543](../../bin/Spaces.py)).
R1 names and formalizes this discipline at the rule-application
boundary specifically, and ties the bivector-category gradient
flow (B6) to the both-branch propagation that supplies its
information.

**Implementation milestone.**  Land alongside Step 6 (grammar
dispatch).  The dispatcher's `apply_rule` becomes
`apply_rules_soft` / `apply_rules_hard` per training vs. inference
mode; the rule predictor's softmax temperature is a config knob
in `<HardPairExtraction>`'s sibling `<RuleSuperposition>` block:

```xml
<RuleSuperposition>
    <softmaxTemperature>1.0</softmaxTemperature>
    <annealRate>0.0</annealRate>     <!-- 0 = no anneal -->
    <hardAtInference>true</hardAtInference>
</RuleSuperposition>
```

---

## Follow-up resolutions (Q1–Q5)

### Q1 — Codebook sharing across spaces

DECIDED — **codebooks do not share**.  PerceptualSpace,
ConceptualSpace, and SymbolicSpace each maintain their own
independent codebook.  No cross-codebook row sharing, no joint
inventory.  The Pi / Sigma layers are the canonical bridge
between them — they project, they don't share rows.

This means each space's `Basis.getW()` returns a private weight
matrix; growth, pruning, and VQ updates are independent per space.
ImpenetrableLayer regularizers operate per-space, on the local
codebook only.

### Q2 — Percept promotion to bivector

DECIDED — **InputSpace emits bytes as vectors in `[-1, +1]`
(bitonic)**, and the bivector promotion at PerceptualSpace ingestion
uses the canonical bitonic-to-bivector map:

```
aP = max(0, x)
aN = max(0, -x)
```

Under the BPE chunking pipeline ([plans/2026-04-23-perceptualspace-bpe-chunking.md](../plans/2026-04-23-perceptualspace-bpe-chunking.md)),
bytes pool into small per-percept codebook entries upstream of
PerceptualSpace; the pooled percept activations are still in the
bitonic `[-1, 1]` range when they reach PerceptualSpace's
ingestion point, so the same `(max(0, x), max(0, -x))` mapping
applies.  No special handling needed for the BPE-pooled case.

### Q3 — N-ary grammar productions: fold order is per-rule

DECIDED — **the fold direction is determined by the rule itself**,
not by a global convention.

Because the new grammar form is `LHS = op(arg1, arg2)` (binary by
default), ternary or higher productions must be written as
**nested calls** that explicitly encode the fold order:

```
S = lift(NP, lift(NP, VP))     # right-fold
S = lift(lift(NP, NP), VP)     # left-fold
```

The grammar author chooses the nesting per rule; the dispatcher
honors what is written without adding its own associativity
convention.

**Loader implications.**  The grammar parser at
[plan Step 6 §2 *Extend the grammar loader*](../plans/2026-04-24-lift-lower-bivector-refactor.md#step-6-grammar-driven-dispatch-via-datagrammarcfg)
needs to handle nested op calls in RHS — recursively parse the
RHS as an expression tree and execute it bottom-up at parse time.
Each internal node of the expression is itself a binary call to
`Ops.<op>`; the leaves are state references.  The runtime
expression tree replaces the flat `[arg1, arg2, ...]` arg list in
the simpler binary case.

This sidesteps `_fold_n_ary` entirely — there is no n-ary fold;
there are only nested binary calls whose order is the author's
commitment.

### Q4 — `VO` deserves its own category vector

DECIDED.  VO (verb-object, introduced as `VO = intersection(VP,
NP)`) gets its own row in `WordSpace.category_codebook` alongside
S, NP, VP, AP, MP, PP, DEF, HAS, and the closed-class terminals.

**Why it matters.**  When the parser reaches a state typed as VO,
the candidate-rule set for the next round is sharply pruned: of all
the productions in `data/grammar.cfg`, only `S = lift(NP, VO)`
takes a VO as an argument.  A committed VO state therefore funnels
the next derivation step into a single deterministic continuation
— a useful constraint that gives the soft-superposition machinery
a strong prior at the next layer.

Without a dedicated category vector, VO would be confused with VP
(its closest sibling), and the rule predictor would distribute
weight across all VP-consuming rules instead of converging on the
unique VO-consuming rule.

**Implementation.**  Add `"VO"` to the enumerated category tag set
in `WordSpace.__init__`'s category-row reservation (per B6's
revised codebook layout).  No other code change — the bivector
form already supports the graded commitment that lets the parser
*move toward* a VO commitment over training.

### Q5 — Deprecation aliases stay, marked with `# XXX`

DECIDED — the legacy `Ops.lift(left, right)` (positional, old
elementwise-product body) and `Ops.lower(left, right)` (positional,
old arithmetic-mean body) deprecation aliases stay in place
indefinitely, with an `# XXX` comment marking each as a manual
follow-up reminder rather than a scheduled removal.

**Rationale.**  Removing aliases requires a coordinated sweep of
every caller, which is more work than the user wants to schedule
right now.  Marking with `# XXX` (the agreed manual-reminder
convention) lets the deprecation warning continue firing while the
user decides when to do the sweep.

**Concrete commit.**

```python
@staticmethod
def lift(left, right):  # XXX deprecated alias — review when convenient
    warnings.warn("Ops.lift(left, right) is the *analysis* product; "
                  "use Ops.lower(x, y, mode='AND') for the synthesis / "
                  "analysis polarity.",
                  DeprecationWarning, stacklevel=2)
    return Ops.lower(left, right, mode='AND')

@staticmethod
def lower(left, right):  # XXX deprecated alias — review when convenient
    warnings.warn("Ops.lower(left, right) is the *synthesis* mean; "
                  "use Ops.lift(x, y, mode='OR') for the synthesis / "
                  "analysis polarity.",
                  DeprecationWarning, stacklevel=2)
    return Ops.lift(left, right, mode='OR')
```

The `# XXX` markers are searchable for a future cleanup pass; no
release-window dependency.

---

## What still needs your call

All 10 O-questions and 7 B-questions are marked DECIDED above.
None are tagged NEEDS-USER-CALL.  If any of the resolutions look
wrong, the most likely candidates to revisit are:

- **O8** (NP coordination semantics) — the propositional reading is
  defensible but reverses my earlier OR/lift classification; if
  empirical sentences favor the union reading, the grammar's
  `conjunction(NP, NP)` should be rewritten to `disjunction(NP,
  NP)`.
- **O7/O9** (`scale` op) — the saturating-multiplication body is one
  of several reasonable choices; if linguistic tests reveal it
  over-saturates or under-modifies, alternatives are
  `lower(DEG, AP, mode='AND')` (deflationary) or a
  parameter-blended hybrid.
- **O5** (`query` body) — the truth-axis-zeroing approach is one of
  several ways to mark a question; if the where-encoding doesn't
  have a free dim for the query bit, an alternative is to allocate
  a dedicated query channel.
- **O6** (`bind` Phase 1) — using `intersection` with side-channel
  head-marking works for now but tightly couples head-extraction to
  state-side metadata rather than to the op itself; if grammatical
  agreement (subject-verb, head-modifier) becomes a primary task,
  promoting `bind` to Phase 1 is cheap and cleaner.
- **B3** (PerceptualSpace bivector) — adopting the same bivector
  shape as ConceptualSpace doubles percept storage.  If percept N
  becomes large (after BPE chunking), revisit whether to keep the
  scalar percept activation and promote to bivector only at the C
  boundary.
- **B7** (layout choice) — leading-bivector vs paired-index.  The
  spec commits to leading-bivector for `subspace.what` and paired-
  index for `TruthLayer` storage, with explicit slicing at the
  boundary.  If the slicing contracts get out of sync, a
  layout-unifying refactor would simplify (but require more
  invasive changes to TruthLayer).

The rest (O1, O2, O3, O4, O10, B1, B2, B4, B5, B6) are mechanical
/ scheduling choices with low downside if revised later.