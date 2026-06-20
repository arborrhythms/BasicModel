# Restoring XOR\_exact: invertible chain + butterfly sigma

Date: 2026-06-04

## Problem

`data/XOR_exact.xml` stopped solving XOR and stopped reconstructing its
inputs, and **no test caught it**.

## Root causes (two layers)

### 1. The test-gap (why nobody noticed)

`test/test_explicit_dimensions.py::TestXorExactCliReconstruction` was marked
`@pytest.mark.xfail`, and there is no `xfail_strict`, so the failure was
silently swallowed (reported `xfailed`, suite stayed green). Its stated reason
(a `bivectorOutput` / `ProjectionBasis` regime) was already stale.

### 2. The convergence regression (what broke)

`git` bisect of intent: `41d44d9` (2026-05-29) is titled *"... XOR\_exact
working again ... All tests passing."* HEAD `e5db52e` (2026-06-04) is 10
commits later — the **modality re-architecture** (which created the new
`bin/architecture.py`). Comparing the working `41d44d9` config to HEAD:

| setting | working `41d44d9` | regressed HEAD |
|---|---|---|
| PS/CS/SS `nDim` | 10 | 14 (where=2/when=2 muxed) |
| **PS `codebook`** | **none** | **quantize** (mandatory) |

The working XOR was a fully **invertible, non-quantized chain**:
embedding $\to$ `PS.pi` (butterfly, codebook=none) $\to$ CS bookkeeping
(codebook=none) $\to$ SS (codebook=none) $\to$ OS. Reverse is the exact
inverse $\Rightarrow$ exact word reconstruction, and XOR is learnable through
the invertible butterfly `pi`.

The modality re-architecture inserted a **mandatory lossy VQ snap at PS**:
`pi` output is snapped to 1-of-1000 prototypes. That snap (a) destroys
per-slot identity so exact reconstruction is impossible (reconstruction
collapsed to a constant), and (b) blocked gradient to `PS.pi` / the embedding
(no straight-through estimator), so XOR sat at chance ($\approx 0.5$).

## Fix

Restore the invertible chain and give the symbolic tier cross-slot reach.

### Config (`data/XOR_exact.xml`)

* PS / CS / SS `codebook=none` — full-width invertible passthrough.
* `butterfly=true` on PS (`pi`) **and** SS (`sigma`): a per-slot fold cannot
  combine the two word slots; XOR needs cross-slot reach at both tiers.
* `lexer=word`, PS `chunking=lexicon`; LR 0.005, 600 epochs.

### Code

* `bin/architecture.py`: `MANDATORY_CODEBOOK_TIERS = set()` — reverted the
  mandatory-PS/SS-codebook constraint so `codebook=none` is allowed.
* `bin/Spaces.py`:
  * `SymbolicSpace` owns `self.sigma` (invertible `SigmaLayer`); the
    `_attach_per_space_syntactic_layer` hook already reads
    `getattr(space, 'sigma')`, so this also revives the default
    `S = sigma(S)` rule (previously a silent no-op).
  * Wired the SymbolicSpace `<butterfly>` flag to the `SigmaLayer`
    constructor ($N = \text{inputShape}[0] \times \text{nOutputDim}$), mirroring
    the `PerceptualSpace.pi` wiring; the per-pair LDU keeps `sigma.reverse`
    exact so the reconstruction round-trip still closes.
  * Removed `ConceptualSpace.sigma_in` / `sigma_cs` and the matching
    `CS.reverse` folds: forward applied none, but reverse applied
    `sigma_in.reverse` unconditionally — an unmatched inverse that corrupted
    reconstruction.
  * `PerceptualSpace` object codebook built with `STE=True` (gradient through
    the VQ snap). Moot for XOR (codebook=none) but corrects any quantize
    config; forward value is unchanged.
* `bin/Models.py`: the non-grammar parallel forward is now perception
  ($PS \to CS_0$) followed by `conceptualOrder` symbolic `sigma` steps
  ($CS_k \to CS_{k+1}$, `_symbolic_sigma_step`); `_reverse_body` inverts each
  `sigma` before `CS.reverse`.
* Tests: removed the `xfail`; updated `test_canonical_shape` and
  `test_modality_codebook` to the reverted (non-mandatory) policy.

## Verification

* `python bin/Models.py data/XOR_exact.xml`: **4/4 OK** word reconstruction;
  predictions $\approx$ labels (`-0.006, 1.015, 1.010, -0.009`); output loss
  $\approx 0.0004$, reconstruction loss $\approx 0.002$.
* `test_modality_configs` build-all: 33/33.
* `test_canonical_shape`, `test_modality_codebook`: green.

## Open items

* `ConceptualSpace` `<butterfly>true</butterfly>` is currently **unwired** —
  CS is a bookkeeping carrier with no `pi` / `sigma` to apply it to (per the
  model the Pi path is PS, the Sigma path is SS). Harmless (inert flag), but
  either wire a CS transform or drop the flag.
* `MANDATORY_CODEBOOK_TIERS = set()` reverts a deliberate modality-architecture
  decision globally. Other configs still opt into codebooks explicitly, so
  this only *allows* `none`; confirm this is the intended scope.
