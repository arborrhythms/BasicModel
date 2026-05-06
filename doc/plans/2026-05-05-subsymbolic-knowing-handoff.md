# Subsymbolic Loop & SubsymbolicSpace — Design Spec

## Overview

This spec adds a second re-entrant loop around `ConceptualSpace`,
parallel to the existing Conceptual↔Symbolic loop, by introducing:

1. A new `SubsymbolicSpace` — same role-shape as `SymbolicSpace`
   (its own muxed event tensor with `.what / .where / .when`) but
   bitonic, codebook-free, and not snapshot to LTM. Its **only
   operations** are an existing-class `SigmaLayer` followed by an
   existing-class `PiLayer`, applied sequentially. No custom
   bridge-layer class is introduced.
2. A **Model-level input combination** that feeds `ConceptualSpace`:

   ```
   conceptual_input  =  perceptual_event  ||  ( symbolic_event + subsymbolic_event )
   ```

   `||` is concatenation along the feature axis; `+` is elementwise
   summation. The summed half is the "internal-percept" contribution
   — the model's own re-entrant view of itself. Symbolic↔subsymbolic
   interference happens architecturally at this summation site,
   not on `PerceptualSpace`.

After this change, `ConceptualSpace`'s input PiLayer reads the
concatenation of (a) the perceptual event tensor and (b) the sum of
the two re-entrant Spaces' event tensors. No new channels are added
to `PerceptualSpace`; its `.what / .where / .when` layout is
unchanged.

> **Architectural note.** An earlier draft proposed renaming a
> hypothetical `.words` per-position channel on `PerceptualSpace`
> to `.thought` and adding a sibling `.knowing` channel. Code
> audit shows no such percept-level channel exists today —
> `SymbolicSpace`'s symbolic content is already carried on its own
> `.what` field on its own muxed event tensor (see
> `Spaces.py:7388–7400`). This spec reflects that finding: the
> two re-entrant Spaces each have their own complete event
> tensors, and combination happens at the Model level.

## Motivation

### Two re-entrant loops, not one

Human cognition shows **dual coding** (Paivio): a verbal /
propositional system and a non-verbal imagery system, both grounded
in concepts and both able to re-enter perception. The current
architecture exposes only the verbal loop (Conceptual↔Symbolic).
The Subsymbolic loop adds the imagistic / embodied counterpart.

### Buddhist Abhidhamma alignment

The six sense-bases of Buddhist psychology treat *mano* (mind) as a
sense whose objects are mental content. Within mano's objects, two
distinguishable streams are recognised:

- *vitakka* / *vikalpa* — verbal thought (Symbolic loop).
- *nimitta* / *pratibimba* — mental image / felt sense
  (Subsymbolic loop).

Each citta-kkhana (moment of consciousness) takes a single unified
input — a co-arising of factors (cetasikas) — and produces one
discerning act. This matches the Model-level combined input per
conceptual order: one moment, one bound input tensor, possibly
carrying sensation and re-entrant contributions simultaneously.

### Integration, not just interference

When the right half (`symbolic_event + subsymbolic_event`) is
zero, `PiLayer`'s `(1+x)/(1-x)` entry transform maps it to the
multiplicative identity (`log = 0`); the right half contributes
nothing and `ConceptualSpace`'s forward is computationally
equivalent to today's perceptual-only forward (verified at
`Layers.py:3286`). When only one re-entrant Space contributes, the
sum equals that contribution alone — pure pass-through at the
summation site. **Interference happens only when both contribute
non-zero values**, and the dynamics it produces depend on whether
the contributions reinforce (integration) or conflict
(interference).

The model is free to learn — under prediction loss — any mixture
along this spectrum:

- bare perception (both re-entrant Spaces zero)
- pure inner monologue (perceptual zero, only Symbolic active)
- pure imagery (perceptual zero, only Subsymbolic active)
- any waking mixture

## Architecture

### Per-Space event tensors (existing pattern)

Each Space carries a muxed event tensor `[B, N, muxedSize]` with
`.what / .where / .when` channels. The pattern is unchanged.

| Space               | Role                                   | What is in `.what`                              |
|---------------------|----------------------------------------|-------------------------------------------------|
| `PerceptualSpace`   | Bottom-up sensation                    | Perceptual embedding (BPE / lexicon prototype)  |
| `SymbolicSpace`     | Linguistic / propositional re-entry    | Bivector codebook content (existing)            |
| `SubsymbolicSpace`  | Imagistic / felt-sense re-entry (new)  | Continuous bitonic activation (no codebook)     |

### Constraint: shared `nDim`, configurable `nVectors`

`nDim` (per-vector dimensionality) is equal across
`PerceptualSpace`, `SymbolicSpace`, and `SubsymbolicSpace`. This
keeps Pi/Sigma bridge layers shape-stable.

`nVectors` is independent per Space, set in XML, **with one
constraint:** `SymbolicSpace.nVectors == SubsymbolicSpace.nVectors`,
because their event tensors are summed elementwise. The Model
config validator enforces this when `SubsymbolicSpace` is enabled.

### `SubsymbolicSpace`

A new `Space` subclass parallel to `SymbolicSpace`:

| Property             | `SymbolicSpace`                      | `SubsymbolicSpace`                              |
|----------------------|--------------------------------------|-------------------------------------------------|
| Codebook on `.what`  | yes (monotonic prototypes)           | **no** (continuous bitonic)                     |
| Domain               | `[0, 1]` magnitude                   | `[-1, 1]` bitonic                               |
| Lattice ops          | conjunction = min, disjunction = max | implicit via `SigmaLayer` (∨) and `PiLayer` (∧) |
| `accumulateTruth`    | yes (LTM via TruthLayer)             | **no** (transient working state)                |
| Resolve / lift       | yes (codebook snap, DoT)             | **no**                                          |
| Internal operations  | resolve / lift / codebook snap       | `SigmaLayer` then `PiLayer`, period             |

**`SubsymbolicSpace`'s only operations are `SigmaLayer` then
`PiLayer`, applied sequentially.** Both are existing classes
(`Layers.py:1737`, `Layers.py:3270`) — there is no custom
`SubsymbolicLayer`. Concretely:

```
SubsymbolicSpace.forward(act_from_conceptual):
    y = self.sigma(act_from_conceptual)        # OR-fold (union)
    z = self.pi(y)                             # AND-fold (conjunction)
    self.subspace.set_event(z)                 # populate .event / .what
    return z

SubsymbolicSpace.reverse(event):
    y = self.pi.reverse(event)
    x = self.sigma.reverse(y)
    return x
```

Both layers are constructed with `invertible=True, monotonic=False`
(bitonic, no codebook constraint). Round-trip invertibility holds
because each layer is individually invertible via the LDU /
triangular-solve path. The lattice intuition is that `Sigma` opens
the input under the union operation, then `Pi` folds it under
conjunction — a one-step ∨-then-∧ composition is the entire
processing pipeline.

This is the literal realisation of the "Sigma-Pi loop" framing
from design discussion: `SubsymbolicSpace`'s body *is* a Sigma·Pi
composition, applied in parallel across conceptual orders to
produce next-order subsymbolic input.

### Model-level combination at `ConceptualSpace` input

The `Model` orchestrates the conceptual order loop. At each order
`t`:

```
conceptual_input_t  =  perceptual_event_t   ||   ( symbolic_event_{t-1} + subsymbolic_event_{t-1} )
```

`ConceptualSpace`'s input PiLayer reads this combined tensor. The
PiLayer's `nInput` widens from today's `muxedSize_p` to
`muxedSize_p + muxedSize_s` (where `muxedSize_s == muxedSize_ss`
per the shared-shape constraint).

When the right half is zero (sentence start, or one of the two
re-entrant Spaces contributing zero), the PiLayer recovers
today's forward exactly because zero is the multiplicative
identity in the entry transform.

### Two re-entrant loops with different timing

- **Symbolic loop (serial within-order).** `ConceptualSpace_t`'s
  PiLayer projects to `SymbolicSpace`; `set_event(act)` populates
  `SymbolicSpace`'s event tensor; resolve / lift / codebook snap
  run within the same conceptual order `t`. Downstream
  consumers within order `t` (output prediction) read the
  finalised `symbolic_event_t`. The same `symbolic_event_t`
  contributes to `conceptual_input_{t+1}`.
- **Subsymbolic loop (parallel across-orders).**
  `ConceptualSpace_t` projects to `SubsymbolicSpace` via the new
  bridge; `subsymbolic_event_t` is computed but only consumed at
  order `t+1`'s combined input. There is no within-order
  consumer at order `t`.

Both contributions feed the next order's combined input
additively. The symbolic contribution is *also* consumed within
the producing order (giving it the "serial" character); the
subsymbolic contribution is only consumed at the next order
("parallel").

### Pass-by-pass dataflow (within one sentence)

```
Pass 0  (pure sensation -- right half zero):
  perceptual_event_0     <-  InputSpace + PerceptualSpace
  symbolic_event_{-1}    =   0   (sentence boundary clear)
  subsymbolic_event_{-1} =   0   (sentence boundary clear)
  conceptual_input_0     =   perceptual_event_0 || 0
  ConceptualSpace.forward(conceptual_input_0)  ->  C_0
  C_0  ->  Pi  ->  SymbolicSpace.set_event(...)     ->  symbolic_event_0
  C_0  ->  Pi  ->  SubsymbolicSpace.set_event(...)  ->  subsymbolic_event_0

Pass 1:
  perceptual_event_1     <-  InputSpace + PerceptualSpace
  conceptual_input_1     =   perceptual_event_1 || ( symbolic_event_0 + subsymbolic_event_0 )
  ConceptualSpace.forward(conceptual_input_1)  ->  C_1
  ...

Pass N:  symmetric.
```

### Sentence boundary

Per locked decision: at sentence start,
`symbolic_event := 0` and `subsymbolic_event := 0`. The
`Model.forward()` orchestration calls `set_event(zero_tensor)` on
both Spaces (or equivalent) before the first conceptual order of
each sentence.

A future config knob (`carryAcrossSentences`, default `false`) is
reserved for long-context working-memory experiments.

## Phasing

### Phase 1 (this spec, target implementation)

The model runs in exactly one of two modes per training / inference
session, configured in XML:

- **`mode = grammar`** — Symbolic loop active. `SubsymbolicSpace`
  exists but its event tensor is held at zero. Combined input
  right half = `symbolic_event + 0 = symbolic_event`.
- **`mode = parallel`** — Subsymbolic loop active.
  `SymbolicSpace`'s event tensor is held at zero (downstream
  symbolic consumers run on the zero tensor; the resolve / lift
  paths skip). Combined input right half = `0 + subsymbolic_event
  = subsymbolic_event`.

In Phase 1 the right half never carries true summation, only one
non-zero contribution. This validates each loop independently
before introducing joint dynamics.

### Phase 2 (deferred)

Both loops active simultaneously. Right half =
`symbolic_event + subsymbolic_event`; the model learns when each
contribution should reinforce, suppress, or override the other.
Manasikāra gating and papañca regularisation become live design
concerns.

## Implementation steps

### 1. Implement `SubsymbolicSpace`

- New subclass of `Space` in `Spaces.py`, alongside `SymbolicSpace`.
- `_build_what_basis()` returns a `Tensor` basis (or `None`); no
  `Codebook`.
- `use_dot_product = False` (bitonic, like `ConceptualSpace`).
- `accumulateTruth = False`. No TruthLayer interaction.
- No resolve / lift / codebook snap.
- **Members:** `self.sigma = SigmaLayer(nDim, nDim,
  invertible=True, monotonic=False, nonlinear=...)` and
  `self.pi = PiLayer(nDim, nDim, invertible=True,
  monotonic=False, nonlinear=...)`. Construct both in
  `__init__` (no defensive `getattr` patterns — initialise
  directly).
- `forward(act)`: `z = self.pi(self.sigma(act))`; then
  `self.subspace.set_event(z)`.
- `reverse(event)`:
  `self.sigma.reverse(self.pi.reverse(event))`.

No new layer class is introduced; the Sigma·Pi composition
inherited from existing layers is the entire processing pipeline.

### 2. Wire the parallel loop in `Model`

- `Model.__init__` constructs `self.subsymbolicSpace` alongside
  `self.symbolicSpace`. Add it to `self.spaces`.
- `Model.forward()` (the conceptual order loop):
  - At each order, before `ConceptualSpace.forward()`, build
    `conceptual_input = perceptual_event || (symbolic_event +
    subsymbolic_event)`. The right-half summation is unconditional
    in code; mode gating zeroes one operand instead of branching.
  - After `ConceptualSpace.forward()`, run *both* Pi projections:
    `Conceptual → SymbolicSpace` (existing) and
    `Conceptual → SubsymbolicSpace` (new).
- Sentence-boundary hook: `symbolicSpace.set_event(zeros_like(...))`
  and `subsymbolicSpace.set_event(zeros_like(...))` before the
  first conceptual order of each sentence.

### 3. Widen `ConceptualSpace`'s input PiLayer

- `ConceptualSpace`'s incoming PiLayer's `nInput` grows from
  `muxedSize_p` to `muxedSize_p + muxedSize_s`.
- Init scheme: existing weights occupy the left
  (`perceptual_event`) half; new weights for the right half
  initialise small / near-zero so that, before training,
  `ConceptualSpace`'s response to a non-zero right half is small
  (the model learns to use the re-entrant signal rather than
  having it as a strong prior). The PiLayer's `x=0 → identity`
  property already gives backward-compatibility when the right
  half is zero; near-zero init on the right-half weights gives
  graceful recovery when it isn't.

### 4. Phase-1 mode gating

- Read `mode` from XML (`grammar` or `parallel`).
- `grammar`: a `subsymbolicEnabled = false` flag causes
  `SubsymbolicSpace`'s output to be held at zero (don't run its
  Pi or set_event; simply leave its event tensor zero).
- `parallel`: a `symbolicEnabled = false` flag analogous to the
  above. SymbolicSpace's existing resolve / lift / codebook paths
  are skipped; its event tensor remains zero.
- The combined-input summation is a single elementwise add either
  way; no branching needed at that line.

### 5. Tests

**Phase 1 (in scope):**

- `test_subsymbolic_sigma_pi_invertibility`: `SubsymbolicSpace`'s
  `forward(reverse(x)) ≈ x` and `reverse(forward(x)) ≈ x` to
  `atol=1e-5` over random `[-1, 1]^nDim` inputs. Verifies the
  Sigma·Pi composition round-trips exactly via each layer's LDU
  inverse.
- `test_subsymbolic_layer_identity`: `SubsymbolicSpace` exposes
  exactly `self.sigma` (a `SigmaLayer`) and `self.pi` (a
  `PiLayer`) as its bridge members; no other custom layer
  classes are introduced.
- `test_combined_input_layout`: at conceptual order `t > 0`,
  `conceptual_input` shape equals
  `[B, N, muxedSize_p + muxedSize_s]`; left half equals
  `perceptual_event_t`; right half equals
  `symbolic_event_{t-1} + subsymbolic_event_{t-1}`.
- `test_zero_right_half_passthrough`: at order 0 (right half
  zero), `ConceptualSpace.forward(conceptual_input_0)` equals
  `ConceptualSpace.forward(perceptual_event_0)` of the
  pre-change model to `atol=1e-5`. Confirms the
  `x=0 → multiplicative identity` property preserves
  backward-compatibility.
- `test_grammar_mode_subsymbolic_zero`: in `mode = grammar`,
  `SubsymbolicSpace.event` is zero at every order. Right-half
  contribution equals `symbolic_event` exactly.
- `test_parallel_mode_symbolic_zero`: in `mode = parallel`,
  `SymbolicSpace.event` is zero at every order. Right-half
  contribution equals `subsymbolic_event` exactly.
- `test_sentence_boundary_reset`: at the first conceptual order
  after a sentence boundary, both Spaces' event tensors are zero
  regardless of mode.
- `test_subsymbolic_one_order_delay`: at order `t > 0`,
  `subsymbolic_event` consumed in the combined input equals
  `SubsymbolicSpace`'s event tensor as written at order `t-1`.
- `test_grammar_mode_baseline_unchanged`: end-to-end metric on a
  small fixture is numerically identical (`atol=1e-5`) to the
  pre-change baseline when `mode = grammar` and
  `subsymbolicEnabled = false`. Regression gate.
- `test_nDim_constraint_validator`: model XML with mismatched
  per-Space `nDim` raises a config-validation error before
  construction.
- `test_nVectors_match_symbolic_subsymbolic`: model XML with
  `SymbolicSpace.nVectors != SubsymbolicSpace.nVectors` raises a
  config-validation error.

**Phase 2 (deferred):**

- `test_combined_input_summation`: with both modes active,
  right half equals `symbolic_event + subsymbolic_event` exactly.
- `test_integration_vs_interference`: aligned contributions
  raise the conceptual activation norm / luminosity; mis-aligned
  contributions reduce it.

## Decision log

Resolved decisions from design discussion:

1. **Combination shape — RESOLVED.**
   `perceptual_event || (symbolic_event + subsymbolic_event)`.
   External half is concatenated; the two re-entrant Spaces' event
   tensors sum into a shared right half. The summation is the
   architectural locus of symbolic↔subsymbolic interference.
2. **`nDim` constraint — RESOLVED.** Enforced as an XML model-file
   requirement. `SymbolicSpace.nVectors == SubsymbolicSpace.nVectors`
   additionally required (for elementwise sum). Both checked in the
   config validator.
3. **Pass-0 init — RESOLVED.** Zero on both re-entrant Spaces.
   Beginner's-mind default; learned prior reserved as future work.
4. **Phase-1 mode exclusivity — RESOLVED.** `grammar` xor
   `parallel`; both-active dynamics deferred to Phase 2.
5. **`SubsymbolicSpace` internal layers — RESOLVED.** No new
   layer class. `SubsymbolicSpace`'s only operations are an
   existing `SigmaLayer` (OR-fold / union) followed by an
   existing `PiLayer` (AND-fold / conjunction), both constructed
   `invertible=True, monotonic=False`. The Sigma·Pi composition
   is the entire bridge. Round-trip invertibility follows from
   each layer's individual LDU inverse.
6. **Per-channel decomposition of the right half — RESOLVED with
   caveat.** Once `symbolic_event + subsymbolic_event` is computed,
   the two operands are not recoverable from the sum. **This is
   intentional**: the operands are recovered by going *up*
   (re-reading `SymbolicSpace.event` and
   `SubsymbolicSpace.event`), not by decomposing the combined
   input. Invertibility holds at each Space's event-tensor level
   independently and at the combined-input level; it does not
   hold at the sum-decomposition level. In Phase 1 the question
   is moot (one operand is always zero).

Deferred (reserved for follow-up plans):

7. **Manasikāra / attention gating + papañca regulariser.**
   Spec assumes uniform read of the combined input via the
   widened PiLayer. A learned per-half (or per-Space) attention
   gate could modulate sensation vs. elaboration; a papañca
   regulariser would penalise high right-half magnitude relative
   to left to preserve sensory contact. Both deferred to a Phase-2
   follow-up plan once both loops are independently working.
8. **Cross-loop direct coupling.** Should `SubsymbolicSpace`
   activations ever be lifted to `SymbolicSpace` (verbalisation
   of imagery), or `SymbolicSpace` lowered to `SubsymbolicSpace`
   (imagistic gloss of thought)? Spec keeps the loops coupled
   only through the shared `ConceptualSpace` bottleneck.
   Deferred; design as separable.

## Audit findings (resolved)

- **`.words` channel** — does not exist as a per-position channel
  on `PerceptualSpace`. All grep hits are unrelated (lex modes,
  embedding vocabulary, OOV handling). The earlier memory's
  `.words` reference was about a derivation-history record on
  `ConceptualSpace`'s grammar forward, not a percept channel.
  **No rename step required.**
- **Symbolic content lives on `SymbolicSpace.what`** —
  confirmed at `Spaces.py:7388-7400`: `ConceptualSpace`'s PiLayer
  produces `act`, then `subspace.set_event(act)` populates
  `SymbolicSpace`'s `.event` and demuxes the first `nWhat=2`
  columns into `.what`. `SubsymbolicSpace` follows the same
  pattern.

## Relationship to existing plans

- **Conceptual Introspection handoff** (`2026-05-04`): the
  `area`, `luminosity`, `directPartOf` operations now have a
  natural home — they are computed on regions in
  `SubsymbolicSpace` (where Gaussian regions live) and become
  scalar / vector annotations on its event tensor. The
  "introspective sidecar" of that plan becomes: introspection
  *is* the Subsymbolic loop's content.
- **Resolve / Luminosity handoff** (`2026-05-04`): the Gaussian
  kernel σ underlies area / luminosity / directPartOf
  computations on `SubsymbolicSpace`. The
  `area(A \ B) ≈ 0` parthood derivation noted in design
  discussion is computable on Subsymbolic regions without
  separate grammar machinery.
- **Memory architecture** (STM / LTM): `SubsymbolicSpace.event`
  is *not* LTM — it is transient working imagery that decays
  (sentence-boundary reset). LTM remains TruthLayer; STM remains
  the per-order `concept_states` trace.
