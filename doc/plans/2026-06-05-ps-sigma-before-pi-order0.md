# PS Sigma-before-Pi substrate with conceptualOrder=0

Date: 2026-06-05

## Problem

The current exact-XOR recovery places the parameterized Sigma operator on
`SymbolicSpace` and uses the parallel body loop to apply it after the
perceptual Pi:

```text
Input -> PS.pi -> CS bookkeeping -> SS.sigma -> CS -> Output
```

That restores XOR, but it also keeps the architectural distinction tangled:

* PS owns the perceptual prototype carrier.
* SS owns a Sigma that is being used as a subsymbolic generalization/readout
  operator rather than as a symbolic/codebook operation.
* `<conceptualOrder>` must be positive to run the Sigma step, even when the
  intended computation is only one perceptual/subsymbolic pass.

The intended cognitive picture is different: perceptual concepts should first
be represented as a local atlas of tight prototype coordinates, then composed
or constrained over that atlas. A single centroid over disparate instances is
the wrong default for animal/perceptual categories.

## Proposal

Move the substrate Sigma out of `SymbolicSpace` and into `PerceptualSpace`,
directly before `PerceptualSpace.pi`:

```text
Input -> PS.sigma_pre -> PS.pi -> CS bookkeeping -> Output
```

Then interpret:

```text
conceptualOrder = 0
```

as:

```text
run exactly one PS substrate pass, but no SS/conceptual recurrence
```

This keeps the PS computation available at order 0 while making order count
mean recurrence depth, not the existence of the base perceptual analyzer.

## Operator semantics

The PS-side Sigma should not be a global category-average collapse. It should
be a prototype-atlas / local-centroid map:

```text
sigma_pre(x): percept/input event -> local prototype coordinates
```

Then Pi composes those local coordinates:

```text
pi(sigma_pre(x)): prototype evidence -> conjunctive / constraint features
```

Architecturally:

```text
sigma_pre = "which local perceptual centroids are active?"
pi        = "which compatible combination of those local centroids holds?"
```

That gives the desired interpretation:

* concepts are fields of tight local prototypes;
* atypical or multi-modal categories can keep several centroids;
* the model avoids forming one centroid over disparate sets;
* symbolic/codebook quantization remains optional and explicit in SS.

## Expected loop structure

### conceptualOrder=0

```text
IS.forward
PS.forward = pi(sigma_pre(IS))
CS.forward = bookkeeping / STM push
OS.forward
```

No `SS.forward`, no `SS.sigma`, no codebook snap, and no symbolic round trip.

### conceptualOrder>0

Run the same base PS pass, then run explicit symbolic/conceptual recurrence
only if requested:

```text
base = PS.forward(IS)
for t in range(conceptualOrder):
    SS/grammar/recurrent step
```

This avoids using conceptual order as a proxy for "did the base substrate
fire?"

## Implementation sketch

1. Add `self.sigma_pre` to `PerceptualSpace`.

   Use the same width and butterfly convention currently used by
   `SymbolicSpace.sigma`, but instantiate it in PS. If XOR requires
   cross-slot mixing, the PS-side Sigma must have butterfly reach over the
   flattened `[B, N * D]` content view.

2. Change `PerceptualSpace.forward`.

   Current effective substrate:

   ```text
   x -> pi(x)
   ```

   Proposed substrate:

   ```text
   x -> sigma_pre(x) -> pi(...)
   ```

3. Remove `SymbolicSpace.sigma`.

   SS should keep codebook/projection/quantization/naming responsibilities,
   but no longer own the default parameterized Sigma substrate.

4. Change the parallel body loop.

   The old order step:

   ```text
   CS_sub = _symbolic_sigma_step(ss, CS_sub)
   ```

   should disappear from the order-0/base path. If higher-order symbolic
   recurrence still needs a Sigma-like operation, it should be named and
   scoped separately from the base perceptual Sigma.

5. Split or revise tests around order semantics.

   Existing tests document `order 0 = identity`. Under this proposal,
   order 0 becomes:

   ```text
   identity recurrence depth, but non-identity base analyzer
   ```

   The cleaner long-term split is:

   ```text
   baseAnalyzerPass = always one PS sigma->pi pass
   conceptualOrder  = number of recurrent CS/SS/symbolic passes
   ```

## Will this solve XOR at conceptualOrder=0?

Not automatically.

It should solve XOR at `conceptualOrder=0` if these conditions hold:

* `sigma_pre` is wide enough to produce a local prototype/feature atlas;
* `sigma_pre` has cross-slot reach when XOR depends on two token positions;
* `pi` preserves multiple output features rather than collapsing to a single
  scalar too early;
* the output head can take an affine/readout combination of the post-Pi
  features;
* PS/CS/SS codebooks are disabled or non-snapping on the exact-XOR path.

It probably will not solve XOR if the new path is only:

```text
scalar = pi(sigma_affine(x))
```

because that is a single monomial/constraint-style collapse. A quick local
sanity check with the existing layer classes showed that `sigma(pi(x))`
solves the tanh-free XOR test essentially exactly, while a scalar
`pi(sigma(x))` gets stuck near chance. That result does not rule out the
proposal; it only says the proposed path must keep a vector of post-Pi
features and use a readout head.

The intended XOR-capable form is:

```text
x
-> sigma_pre(x)          # wide local prototype / clause atlas
-> pi(...)               # product / compatibility features
-> OutputSpace readout   # affine combination, e.g. x + y - 2xy
```

With that structure, XOR should be learnable at `conceptualOrder=0` because
the required interaction feature can be produced in PS and the symbolic loop is
not needed.

## Rationale from cognition

This aligns the architecture with the prototype/category-learning picture:

* perceptual categories are often better modeled as local prototype clusters
  than as one centroid over a scattered superordinate;
* multiple prototype/exemplar systems can coexist;
* composition should happen over local perceptual evidence rather than after
  collapsing that evidence into a global symbolic average.

In this reading, `pi(sigma(x))` is not chosen because it preserves Euclidean
convexity in the strict mathematical sense. It does not generally do that.
It is chosen because it preserves a local-prototype atlas before composition,
which is a better cognitive fit for perceptual concepts.

## Risks

* This changes `<conceptualOrder>` semantics and will require test updates.
* If `sigma_pre` is implemented as a narrow/global affine layer, it can erase
  the very cluster structure the proposal depends on.
* Removing `SS.sigma` may break existing reconstruction paths that currently
  expect `_reverse_body` to invert symbolic Sigma steps.
* The proposal separates symbolic recurrence from the base analyzer, so code
  paths that assume `len(symbolicSpaces) == conceptualOrder` may need cleanup
  for order 0.

## Acceptance criteria

* `XOR_exact.xml` or an order-0 variant solves XOR with `conceptualOrder=0`.
* Exact-XOR reconstruction remains 4/4 when codebooks are disabled.
* Serial mode does not quantize SS on ordinary round trips.
* `conceptualOrder=0` runs PS sigma-before-pi once and skips SS recurrence.
* `conceptualOrder>0` still has an explicit, tested symbolic/grammar
  recurrence path.
