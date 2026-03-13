
# Invertible Transformation Based on Hyperbolic Products

## Forward Transformation

Given input vector $x \in \mathbb{R}^n$, parameters $W \in \mathbb{R}^{m \times n}$, and positive constants $b_j > 0$, define:

$$
y_j = b_j \prod_{i=1}^n \left(1 - \tanh(w_{ji} x_i)\right)
$$
$$
z_j = b_j \prod_{i=1}^n \left(1 + \tanh(w_{ji} x_i)\right)
$$

This mapping is invertible under mild assumptions on $W$.

---

## Inverse Transformation

We can recover a latent variable $\gamma_j$ as:

$$
\frac{z_j}{y_j} = \prod_{i=1}^n \frac{1 + \tanh(w_{ji} x_i)}{1 - \tanh(w_{ji} x_i)}
$$

Using the identity:

$$
\frac{1 + \tanh(u)}{1 - \tanh(u)} = e^{2u}
$$

we get:

$$
\frac{z_j}{y_j} = e^{2 \sum_{i=1}^n w_{ji} x_i}
$$

so that:

$$
\gamma_j := \frac{1}{2} \log\left( \frac{z_j}{y_j} \right) = \sum_{i=1}^n w_{ji} x_i
$$

This gives:

$$
\gamma = W x
$$

and if $W$ is invertible:

$$
x = W^{-1} \gamma
$$

or:

$$
x \approx W^{\dagger} \gamma
$$

if $W$ is not square and we use the pseudoinverse.

---

## Log-Determinant of the Jacobian

The log-determinant of the Jacobian is required if this transformation is used in normalizing flows:

$$
\log |\det J| = \sum_{j=1}^m \sum_{i=1}^n \log\left(1 - \tanh^2(w_{ji} x_i)\right)
$$

since:

$$
\frac{d}{dx} \tanh(w_{ji} x_i) = w_{ji} \left(1 - \tanh^2(w_{ji} x_i)\right)
$$

---

# Analysis of $y_j = \tanh\left( \sum_{i=1}^n W_{ji} x_i \right)$

## Forward Transformation

This is a standard activation:

$$
y_j = \tanh\left( \sum_{i=1}^n W_{ji} x_i \right)
$$

or:

$$
y = \tanh(W x)
$$

## Inverse Transformation

The inverse exists coordinate-wise, provided $|y_j| < 1$:

$$
\sum_{i=1}^n W_{ji} x_i = \tanh^{-1}(y_j)
$$

So:

$$
W x = \tanh^{-1}(y)
$$

If $W$ is invertible:

$$
x = W^{-1} \tanh^{-1}(y)
$$

or:

$$
x \approx W^{\dagger} \tanh^{-1}(y)
$$

if $W$ is not square.

## Log-Determinant

The Jacobian is:

$$
\frac{\partial y_j}{\partial x_i} = W_{ji} \left(1 - y_j^2\right)
$$

Thus:

$$
\log |\det J| = \log |\det W| + \sum_{j=1}^m \log(1 - y_j^2)
$$
