# Ergodic Exploration

This document describes the current implementation in `bin/Layers.py`.
The theory-level motivation belongs in [Architecture.md](Architecture.md) and
[MachineMinds.md](MachineMinds.md); this page is the operational contract.

## Effective Weights

`ErgodicLayer` mixes learned structure with sampled noise through two scalar
buffers:

$$
W_{\text{eff}} = \text{bias} \cdot W + \text{var} \cdot \varepsilon
$$

- `bias` is the learned-weight trust.
- `var` is the exploration-noise scale.
- `bias + var` is approximately 1, with clamps for numerical stability.

The buffers are not trained by Adam. They are updated from observed gradient
energy in `paramUpdate()`.

## Initialization

Current initialization is conservative:

- `ErgodicLayer` starts with `bias = 1` and `var = 0`.
- `set_sigma(0.5)` sets the responsiveness parameter, but does not itself make
  the first forward noisy.
- The first nonzero exploration update happens after a backward pass, when
  `paramUpdate()` observes gradients.
- `LinearLayer(ergodic=True)` initializes `W` to an identity matrix.
- `InvertibleLinearLayer` initializes its LDU factors to identity
  (`raw_L = 0`, `d = 1`, `raw_U = 0`).

This supersedes older notes that described alpha-zero startup or zero-initialized
ergodic weights.

## Gradient Energy Sensor

`observe_sigma()` scans learnable parameters with gradients, ignores noise
buffers, and accumulates mean squared gradient energy. It keeps an EMA-style
running variance with:

- `sigma_beta = 0.99`
- `sigma_kappa = 0.01 / requested_sigma` via `set_sigma(sigma)`

`sigma_to_ergodic()` bias-corrects the variance estimate and maps it to the
exploration buffer:

$$
\text{var} = \frac{s}{s + \kappa}, \qquad
\text{bias} = 1 - \text{var}
$$

with implementation clamps:

- `var <= 0.95`
- `bias >= 0.05`

Low observed variance therefore favors exploitation; high observed variance
increases exploration.

## Update Order

In the training path, the ergodic sensor is updated after `backward()` while
gradients are still available:

```python
loss.backward()
if model.ergodic:
    model.paramUpdate()
optimizer.step()
```

`paramUpdate()` is a no-op when `ergodic=False`.

## Manual Control

`set_sigma(sigma)` adjusts responsiveness:

- `sigma == 0` sets `sigma_kappa` very high, effectively suppressing exploration.
- `sigma > 0` sets `sigma_kappa = 0.01 / sigma`.

The code uses this as a control surface for deterministic/evaluation paths and
for experiments that want more or less exploration.

## Layer Details

### `LinearLayer`

When `ergodic=True`, `LinearLayer` stores:

- `W`: learned identity-initialized weight matrix;
- `noise`: sampled matrix buffer;
- optional `biasWeight` and `biasNoise`.

The forward path applies the same `bias * learned + var * noise` pattern to
weights and optional bias.

### `InvertibleLinearLayer`

`InvertibleLinearLayer` represents:

$$
W = L \cdot D_{\text{embed}} \cdot U
$$

Noise is injected into the LDU factors, not into a materialized dense inverse.
The reverse path uses triangular solves, so the inverse remains exact for the
effective factors. No SVD path is used.

### `SigmaLayer` and `PiLayer`

`SigmaLayer` and `PiLayer` use `LinearLayer` or `InvertibleLinearLayer`
internally, so they inherit the same ergodic behavior. Composite layers forward
`set_sigma()` and `paramUpdate()` to their child layers.

## Current Limits

- There is no `global_temp` knob in the current implementation.
- `dropoutRate` exists as a field on `ErgodicLayer`, but the documented
  exploration mechanism is the `bias`/`var` noise mix.
- The implementation is scalar per layer, not a learned per-neuron temperature
  trained by Adam.
