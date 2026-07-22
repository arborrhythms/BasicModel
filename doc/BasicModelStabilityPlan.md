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
