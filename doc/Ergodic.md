# Ergodic Exploration via Adaptive Bias-Variance Control

BasicModel decouples **what** the network learns (weights $W$) from **how
much it explores** (a scalar $\alpha$ governing the bias-variance tradeoff).
Two algorithms operate in tandem:

| Component | Algorithm | Role |
|-----------|-----------|------|
| Weights $W$ | Standard Adam | Gradient descent on the loss landscape |
| Tradeoff $\alpha$ | Gradient Energy Sensor | Measures loss-surface curvature to set exploration |

**$\alpha$ is not trained by gradient descent.** It is a *sensor* reading the
gradient energy through the temperature parameter and converting it into an
exploration policy.

---

## 1. The Ergodic Weights

Each layer uses an effective weight matrix sampled from a locally specified
ergodic distribution:

$$
W_{\text{eff}} = \alpha \cdot W + (1 - \alpha) \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

$W$ stores the learned structure that shapes the distribution as $\alpha$
rises. Derived quantities:

$$
\text{bias} = \alpha, \qquad \text{var} = 1 - \alpha
$$

- $\alpha = 0$: Pure exploration. $W_{\text{eff}} = \varepsilon$. Random noise.
- $\alpha = 1$: Pure exploitation. $W_{\text{eff}} = W$. Learned weights only.
- $0 < \alpha < 1$: Interpolation.

Training begins at $\alpha = 0$ and transitions autonomously toward
exploitation as the loss landscape flattens.

---

## 2. Why Standard Adam Fails on $\alpha$

The temperature $T = 1 - \alpha$ receives gradients via:

$$
\frac{\partial \mathcal{L}}{\partial T} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial (W_{\text{eff}})_{ij}} \cdot \varepsilon_{ij}
$$

Because $\varepsilon$ is re-sampled every forward pass, this gradient is
**zero-mean**:

$$
\mathbb{E}\!\left[\frac{\partial \mathcal{L}}{\partial T}\right] = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial (W_{\text{eff}})_{ij}} \cdot \mathbb{E}[\varepsilon_{ij}] = 0
$$

Standard Adam's first moment kills the signal: $m_t \approx 0 \Rightarrow m_t /
\sqrt{v_t} \approx 0$. Adam produces no update.

---

## 3. The Gradient Energy Sensor

### 3.1 Modified Second-Moment Estimator

Discard the first moment; use only the second moment as **output** rather
than normalizer:

$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2
$$

where $g_t = \partial \mathcal{L} / \partial T$ and $\beta = 0.999$.

### 3.2 Bias Correction

$$
\hat{v}_t = \frac{v_t}{1 - \beta^t}
$$

### 3.3 What $\hat{v}_t$ Measures

The second moment of a zero-mean random variable is its variance:

$$
\mathbb{E}[g_t^2] = \text{Var}(g_t) = \sum_{i,j} \left(\frac{\partial \mathcal{L}}{\partial (W_{\text{eff}})_{ij}}\right)^2 = \left\| \frac{\partial \mathcal{L}}{\partial W_{\text{eff}}} \right\|_F^2
$$

The **squared Frobenius norm** of the loss gradient — the **gradient energy**.

| Gradient energy | Meaning | Desired behavior |
|-----------------|---------|-----------------|
| High $\hat{v}_t$ | Loss sensitive to weight changes | Keep exploring (low $\alpha$) |
| Low $\hat{v}_t$ | Loss surface is flat | Exploit (high $\alpha$) |

### 3.4 The Alpha Update Rule

$$
\alpha_t = \frac{1}{1 + \tau \cdot \sqrt{\hat{v}_t}}
$$

where $\tau$ (`global_temp`) is a sensitivity knob.

Properties:
- $\sqrt{\hat{v}_t} \to 0 \Rightarrow \alpha_t \to 1$ (exploit)
- $\sqrt{\hat{v}_t} \to \infty \Rightarrow \alpha_t \to 0$ (explore)
- $\alpha_t \in (0, 1)$ always
- $\tau$ controls transition speed

After computing $\alpha_t$: $\text{bias}_t = \alpha_t$, $\text{var}_t = 1 -
\alpha_t$.

### 3.5 Layer-Local Certainty (Per-Neuron Sigma)

Each ergodic layer maintains per-neuron **sigma** — running variance of its
gradient energy, tracked via Welford's algorithm in `observe_sigma()`. Sigma
drives per-neuron bias and var:

$$
\text{var}_j = \frac{\sigma_j}{\sigma_j + \kappa}, \qquad \text{bias}_j = 1 - \text{var}_j
$$

Low sigma (consistent gradient, found a minimum) yields high bias, low var.
High sigma (unstable gradient) yields low bias, high var.

`set_sigma(sigma)` controls $\kappa$:
- `sigma=1`: encourage exploration (low $\kappa$)
- `sigma=0`: suppress exploration (high $\kappa$, var $\approx 0$)

---

## 4. Comparison: Adam vs. Gradient Energy Sensor

| Property | Adam (for $W$) | Sensor (for $\alpha$) |
|----------|----------------|------------------------|
| **Type** | Optimizer | Sensor (measurement) |
| **First moment** $m_t$ | Yes — tracks direction | No — direction is zero-mean |
| **Second moment** $v_t$ | Yes — normalizes step size | Yes — **is the output** |
| **Role of** $\sqrt{v_t}$ | Denominator (adaptive LR) | Numerator (energy estimate) |
| **Update rule** | $W \leftarrow W - \eta \cdot m / \sqrt{v}$ | $\alpha \leftarrow 1 / (1 + \tau \sqrt{\hat{v}})$ |
| **Descent?** | Yes | No — maps energy to policy |
| **Zero-mean safe?** | No — produces $0/\sqrt{v} = 0$ | Yes — by design |

---

## 5. Derivation: Why Drop $m_t$

Let $g_t = \nabla_T \mathcal{L}$. Since noise $\varepsilon$ is i.i.d. each
step, $\mathbb{E}[g_t] = 0$. The first moment EMA $m_t = \beta_1 m_{t-1} +
(1 - \beta_1) g_t$ gives $\mathbb{E}[m_t] = \beta_1 \mathbb{E}[m_{t-1}]$. By
induction $\mathbb{E}[m_t] = \beta_1^t \mathbb{E}[m_0] = 0$ — exponentially
to zero, carrying **no information**.

The second moment converges to a meaningful quantity:

$$
\mathbb{E}[v_t] \to \mathbb{E}[g_t^2] = \left\| \nabla_{W_{\text{eff}}} \mathcal{L} \right\|_F^2
$$

The gradient energy — exactly what we need.

---

## 6. Dropout via Bias

`bias` and `var` support dropout:

$$
\text{bias}_t^{\text{(drop)}} = \begin{cases}
0 & \text{with probability } p \\
\alpha_t & \text{with probability } 1 - p
\end{cases}
$$

When bias is dropped to zero, the layer acts as pure noise regardless of
$\alpha$. The temperature $T = 1 - \alpha$ remains unchanged so noise
contribution is consistent.

Dropout is a schedule on bias: ordinary Bernoulli dropout is the binary case.
Annealed dropout corresponds to increasing the expected bias over training.
Nonzero temperature bakes regularization directly into the model — every
forward pass perturbs the effective weights, discouraging brittle
co-adaptation.

---

## 7. The `global_temp` Sensitivity Knob

$\tau$ (`global_temp`) scales responsiveness:

| $\tau$ | Effect |
|--------|--------|
| $\tau = 0$ | $\alpha = 1$ always — pure exploitation |
| $\tau \ll 1$ | Quickly converges to $\alpha \approx 1$ |
| $\tau = 1$ | Balanced — $\alpha$ tracks gradient energy on natural scale |
| $\tau \gg 1$ | Maintains high exploration even with moderate gradients |

$\tau$ can be scheduled (e.g., annealed from high to low) for a coarse
exploration-to-exploitation curriculum.

---

## 8. Initialization and Bootstrap

**Problem.** If initialized at $\alpha = 1$ (pure exploitation), the noise
term vanishes and $\partial \mathcal{L} / \partial T = 0$. The sensor would
read zero energy and keep $\alpha = 1$ forever — an **exploitation trap**.

**Solution.** Initialize at $\alpha = 0$.

At $\alpha = 0$:
- $W_{\text{eff}} = \varepsilon$ — pure noise carries gradients through the network
- Temperature gradient is nonzero: $g_t = \sum_{ij} (\partial \mathcal{L} /
  \partial \varepsilon_{ij}) \cdot \varepsilon_{ij} \neq 0$
- Sensor measures initial gradient energy and begins tuning $\alpha$ upward
- As $W$ learns useful structure, energy decreases and $\alpha$ rises

Zero-initializing $W$ is safe and often cleaner than random initialization.
Symmetry is broken by the sampled noise; learned weights don't contribute at
$\alpha = 0$ anyway. Starting from $W = 0$ makes the exploratory regime
honest.

---

## 9. Layer Architecture

All ergodic layers follow the effective-weight pattern. When `ergodic=True`,
noise buffers are registered and weights are zero-initialized.

### ErgodicLayer Base Class

`ErgodicLayer` (`Model.py`) provides per-neuron `bias` and `var` buffers
driven by gradient variance (`observe_sigma()` / `sigma_to_ergodic()`).
Subclasses with child ErgodicLayers must forward `set_sigma()` and
`paramUpdate()`.

### InvertibleLinearLayer — factor-level noise injection

LDU factorisation $W = L \cdot D_{\text{embed}} \cdot U$ (see
[Architecture.md](Architecture.md)). Exact inverse via triangular solves;
no SVD.

When `ergodic=True`, noise is injected at the factor level to preserve exact
invertibility at all temperatures:

$$
L_{\text{eff}} = I + \text{strict\_lower}(\text{raw\_L} + t \cdot \text{noise\_L})
$$
$$
U_{\text{eff}} = I + \text{strict\_upper}(\text{raw\_U} + t \cdot \text{noise\_U})
$$
$$
d_{\text{eff}} = b \cdot d_{\text{clamped}} + t \cdot \text{noise\_d}
$$

`W_eff` stays in LDU form, so its exact inverse is always available — no
approximation regardless of noise level. Compare to matrix-level blending
$W_{\text{eff}} = b \cdot W + t \cdot N$, which destroys LDU structure:

$$
W_{\text{eff}} \cdot W_{\text{eff}}^{-1} \approx (b^2 + t^2) I + b t (W N^{-1} + N W^{-1}) \neq I
$$

Error grows with temperature — exactly the regime where exploration is most
active. Factor-level injection avoids this.

`noise_d` is sampled with magnitude in $[\text{eps}, 1]$ and random sign, so
it's always invertible; combined with the `stable=True` clamp on the
deterministic diagonal, `d_eff` stays bounded away from zero.

`SigmaLayer(invertible=True)` and `PiLayer(invertible=True)` use
`InvertibleLinearLayer(ergodic=True)` internally. `bias` and `var` from the
parent `ErgodicLayer` forward as `bias=` / `temp=` arguments, so the
gradient-energy sensor governs noise injection in the inner linear
primitive.

---

## 10. Training Loop Integration

```python
optimizer = torch.optim.Adam(model.getParameters(), lr=lr)  # Adam for W
loss = model(x, y)                                          # forward W_eff = bias*W + var*noise
loss.backward()                                             # Adam: dL/dW; sensor: dL/dT
optimizer.step()
model.paramUpdate()                                         # sensor updates alpha
```

---

## 11. Summary of the Full Algorithm

**Initialize:** $\alpha_0 = 0$, $v_0 = 0$, $t = 0$, $\beta = 0.999$.

**Each step:**

1. Sample $\varepsilon \sim \mathcal{N}(0, I)$.
2. $W_{\text{eff}} = \alpha_t W + (1 - \alpha_t) \varepsilon$.
3. Forward: $\hat{y} = f(x; W_{\text{eff}})$; loss $\mathcal{L}(\hat{y}, y)$.
4. Backward: $\nabla_W \mathcal{L}$ and $g_t = \nabla_T \mathcal{L}$.
5. **Adam** updates $W$.
6. **Sensor** updates $\alpha$:

$$
v_t \leftarrow \beta v_{t-1} + (1 - \beta) g_t^2,
\quad
\hat{v}_t = \frac{v_t}{1 - \beta^t},
\quad
\text{var}_j = \frac{\hat{v}_j}{\hat{v}_j + \kappa}, \quad \text{bias}_j = 1 - \text{var}_j
$$

7. Zero temperature gradient: $g_t \leftarrow 0$.
