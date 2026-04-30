# Lexicon

## Overview

The **Lexicon** is the project's vocabulary embedding: a learnable map from
discrete tokens — words, symbols, byte-pair units, or other vocabulary items —
to vectors in a bounded continuous space.

It is implemented as [`Lexicon(nn.Embedding)`](../bin/Layers.py) and lives at
the perceptual / symbolic boundary of the pipeline. Codebook lookup,
nearest-neighbor decode, and reverse-pass reconstruction all pass through the
Lexicon.

The Lexicon's default geometry is now the **unit ball**:

\[
B^D = \{x \in \mathbb{R}^D : \|x\|_2 \le 1\}.
\]

Each vocabulary row is a vector

\[
w_i \in B^D.
\]

This replaces the earlier default flat-torus wrapped-MSE lookup. The motivation
is computational: nearest-neighbor lookup in the unit ball can be expressed as
a dense matrix multiplication plus a precomputed norm correction. That gives an
exact L2 sort while avoiding coordinatewise wrap/modulo operations.

The unit-ball Lexicon has three practical properties:

- **No hypercube corners.** The embedding is bounded by a smooth ball rather
  than by a coordinate-aligned box.
- **Magnitude channel.** Vector norm can represent strength, certainty,
  specificity, or activation magnitude.
- **Fast exact lookup.** Euclidean nearest-neighbor search reduces to

  \[
  \operatorname{score}(x,w_i)=x^\top w_i-\frac12\|w_i\|_2^2,
  \]

  so top-k lookup is a matrix multiplication followed by a bias subtraction.

The unit ball has a real center and boundary. It is therefore not homogeneous in
the same sense as a flat torus. This is an intentional engineering tradeoff:
the Lexicon uses a simpler, faster, corner-free geometry rather than
coordinatewise modular arithmetic.

---

## Geometry

### Unit ball

The Lexicon stores each row inside the closed unit ball:

\[
w_i \in B^D,\qquad \|w_i\|_2 \le 1.
\]

An incoming vector is also projected or constrained to the unit ball:

\[
x \in B^D,\qquad \|x\|_2 \le 1.
\]

The canonical distance is ordinary squared Euclidean distance:

\[
d^2(x,w_i)=\|x-w_i\|_2^2.
\]

Expanding:

\[
\|x-w_i\|_2^2
=
\|x\|_2^2+\|w_i\|_2^2-2x^\top w_i.
\]

For a fixed query \(x\), the term \(\|x\|_2^2\) is constant across all
candidates. Therefore sorting by smallest distance is equivalent to sorting by
largest score:

\[
\operatorname{score}(x,w_i)
=
x^\top w_i-\frac12\|w_i\|_2^2.
\]

This is the central fast lookup identity.

---

## Fast Lookup

### Stored lookup state

For a lexicon matrix

\[
W\in\mathbb{R}^{V\times D},
\]

store:

```python
W_index: Tensor[V, D]   # lexicon rows, projected into the unit ball
W_norm2: Tensor[V]      # precomputed squared row norms
```

where

```python
W_norm2 = W_index.square().sum(dim=-1)
```

The lookup score is:

```python
score = x @ W_index.T - 0.5 * W_norm2
idx = score.topk(k, largest=True).indices
```

This produces the exact same sort as ascending Euclidean squared distance in
the unit ball.

### Why this is faster than wrapped-MSE

The older wrapped-MSE torus lookup used the form:

```python
delta = wrap(W - x)
dist_sq = delta.square().mean(dim=-1)
idx = dist_sq.topk(k, largest=False).indices
```

That requires coordinatewise subtraction, modulo/wrap, squaring, reduction, and
top-k over a large \(B\times V\times D\) comparison.

The unit-ball lookup uses:

```python
score = x @ W.T - 0.5 * W_norm2
idx = score.topk(k, largest=True).indices
```

The expensive part is a dense matrix multiplication, which is highly optimized
on GPUs and can use mixed precision efficiently. The result is still an exact
nearest-neighbor sort for the chosen unit-ball L2 geometry.

---

## Initialization and Maintenance

### Initialization

A unit-ball Lexicon may be initialized by drawing Gaussian vectors and scaling
them into the ball:

```python
W = torch.randn(V, D)
W = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-12)
radius = torch.rand(V, 1).pow(1.0 / D)
W = radius * W
```

This samples directions uniformly and radii with the correct volume weighting
for a uniform distribution in the unit ball.

A simpler initialization is also acceptable:

```python
W = 0.02 * torch.randn(V, D)
W = project_unit_ball(W)
```

### Projection

After optimizer steps, project rows back into the unit ball:

\[
w_i \leftarrow
\begin{cases}
w_i, & \|w_i\|_2 \le 1,\\
w_i / \|w_i\|_2, & \|w_i\|_2 > 1.
\end{cases}
\]

In code:

```python
def project_unit_ball_(W, eps=1e-12):
    with torch.no_grad():
        norm = W.norm(dim=-1, keepdim=True).clamp_min(eps)
        scale = torch.clamp(1.0 / norm, max=1.0)
        W.mul_(scale)
    return W
```

After projection, refresh the lookup state:

```python
W_index = W.contiguous()
W_norm2 = W_index.square().sum(dim=-1).contiguous()
```

---

## PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple


class Lexicon(nn.Embedding):
    """
    Unit-ball vocabulary embedding.

    Rows are constrained to ||w_i|| <= 1. Fast nearest-neighbor lookup under
    Euclidean L2 uses:

        score(x, w_i) = x @ w_i - 0.5 * ||w_i||^2

    Sorting by descending score is exactly equivalent to sorting by ascending
    squared L2 distance for fixed x.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init: str = "uniform_ball",
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self.reset_unit_ball_parameters(init=init)

    @torch.no_grad()
    def reset_unit_ball_parameters(self, init: str = "uniform_ball") -> None:
        V, D = self.weight.shape

        if init == "uniform_ball":
            W = torch.randn_like(self.weight)
            W = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            radius = torch.rand(
                V, 1,
                device=self.weight.device,
                dtype=self.weight.dtype,
            ).pow(1.0 / D)
            self.weight.copy_(radius * W)

        elif init == "small_normal":
            self.weight.normal_(mean=0.0, std=0.02)
            self.project_unit_ball_()

        else:
            raise ValueError(f"unknown init: {init}")

    @staticmethod
    def project_unit_ball(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Return x projected rowwise into the closed unit ball.
        """
        norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        scale = torch.clamp(1.0 / norm, max=1.0)
        return x * scale

    @torch.no_grad()
    def project_unit_ball_(self, eps: float = 1e-12) -> None:
        """
        In-place rowwise projection of the embedding matrix into the unit ball.
        Call after optimizer.step().
        """
        norm = self.weight.norm(dim=-1, keepdim=True).clamp_min(eps)
        scale = torch.clamp(1.0 / norm, max=1.0)
        self.weight.mul_(scale)

    @torch.no_grad()
    def lookup_index(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return lookup tensors:

            W_index: (V, D)
            W_norm2: (V,)

        W_norm2 should be refreshed after every optimizer step that changes
        the lexicon.
        """
        W_index = self.weight.contiguous()
        W_norm2 = W_index.square().sum(dim=-1).contiguous()
        return W_index, W_norm2

    @staticmethod
    def l2_scores(
        x: torch.Tensor,
        W_index: torch.Tensor,
        W_norm2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L2-equivalent lookup scores.

        Args:
            x:
                Query tensor of shape (..., D).

            W_index:
                Lexicon matrix of shape (V, D).

            W_norm2:
                Squared row norms of W_index, shape (V,).

        Returns:
            scores:
                Shape (..., V). Larger means closer under Euclidean L2.
        """
        if x.shape[-1] != W_index.shape[-1]:
            raise ValueError(
                f"x last dim {x.shape[-1]} must equal W_index dim {W_index.shape[-1]}"
            )

        return x @ W_index.T - 0.5 * W_norm2

    @staticmethod
    def topk_l2(
        x: torch.Tensor,
        W_index: torch.Tensor,
        W_norm2: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exact top-k nearest rows under Euclidean L2 in the unit ball.

        Args:
            x:
                Query tensor of shape (..., D).

            W_index:
                Lexicon matrix of shape (V, D).

            W_norm2:
                Squared row norms of W_index, shape (V,).

            k:
                Number of nearest rows to return.

        Returns:
            indices:
                Shape (..., k), nearest row indices.

            dist_sq:
                Shape (..., k), exact squared L2 distances.

            scores:
                Shape (..., k), L2-equivalent scores.
        """
        if k <= 0:
            raise ValueError("k must be positive")

        V, D = W_index.shape
        k = min(k, V)

        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, D)

        x_norm2 = x_flat.square().sum(dim=-1, keepdim=True)
        scores_all = x_flat @ W_index.T - 0.5 * W_norm2.unsqueeze(0)
        scores, indices = torch.topk(scores_all, k=k, dim=-1, largest=True)

        selected_w_norm2 = W_norm2[indices]
        selected_ip = scores + 0.5 * selected_w_norm2
        dist_sq = x_norm2 + selected_w_norm2 - 2.0 * selected_ip

        out_shape = original_shape + (k,)

        return (
            indices.reshape(out_shape),
            dist_sq.reshape(out_shape),
            scores.reshape(out_shape),
        )
```

---

## Chunked Top-k Lookup

For very large vocabularies or large query batches, avoid materializing the full
score matrix \((B,V)\). Use chunked scoring over lexicon rows.

```python
@torch.no_grad()
def topk_l2_chunked(
    x: torch.Tensor,
    W_index: torch.Tensor,
    W_norm2: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 131_072,
):
    """
    Exact chunked top-k nearest lookup under unit-ball Euclidean L2.

    This avoids materializing a full (B, V) score matrix.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    V, D = W_index.shape
    k = min(k, V)

    original_shape = x.shape[:-1]
    x_flat = x.reshape(-1, D)

    best_scores = None
    best_indices = None

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)

        Wc = W_index[start:end]
        Nc = W_norm2[start:end]

        scores_c = x_flat @ Wc.T - 0.5 * Nc.unsqueeze(0)

        local_k = min(k, end - start)
        scores_k, indices_k = torch.topk(
            scores_c,
            k=local_k,
            dim=-1,
            largest=True,
        )
        indices_k = indices_k + start

        if best_scores is None:
            best_scores = scores_k
            best_indices = indices_k
        else:
            merged_scores = torch.cat([best_scores, scores_k], dim=-1)
            merged_indices = torch.cat([best_indices, indices_k], dim=-1)

            best_scores, pos = torch.topk(
                merged_scores,
                k=k,
                dim=-1,
                largest=True,
            )
            best_indices = merged_indices.gather(-1, pos)

    x_norm2 = x_flat.square().sum(dim=-1, keepdim=True)
    Wk_norm2 = W_norm2[best_indices]
    selected_ip = best_scores + 0.5 * Wk_norm2
    dist_sq = x_norm2 + Wk_norm2 - 2.0 * selected_ip

    out_shape = original_shape + (k,)

    return (
        best_indices.reshape(out_shape),
        dist_sq.reshape(out_shape),
        best_scores.reshape(out_shape),
    )
```

---

## Training Pattern

A typical training step should keep the weight matrix in the ball:

```python
loss.backward()
optimizer.step()

with torch.no_grad():
    lexicon.project_unit_ball_()
```

Then refresh the lookup state:

```python
W_index, W_norm2 = lexicon.lookup_index()
```

For mixed precision, keep a stable master weight where appropriate, and compute
lookup scores using the dtype appropriate for the hardware. The lookup is
matmul-heavy and therefore works well with `float16`, `bfloat16`, or AMP,
depending on the target device.

---

## API Summary

```python
lexicon = Lexicon(V, D)

# Training maintenance
lexicon.project_unit_ball_()

# Lookup index
W_index, W_norm2 = lexicon.lookup_index()

# Exact top-k L2 lookup
idx, dist_sq, scores = Lexicon.topk_l2(x, W_index, W_norm2, k=32)

# Chunked exact top-k L2 lookup
idx, dist_sq, scores = topk_l2_chunked(
    x,
    W_index,
    W_norm2,
    k=32,
    chunk_size=131_072,
)
```

---

## Geometric Facts

- **Bounded.** Every row lies inside the unit ball.
- **No corners.** The boundary is smooth rather than coordinate-aligned.
- **Norm is meaningful.** Vector norm may encode strength, confidence,
  specificity, or activation magnitude.
- **Has a center.** The origin is a real distinguished point. This is unlike
  the flat torus, which has no preferred origin.
- **Has a boundary.** Rows may approach the unit sphere. Projection prevents
  optimizer drift outside the ball.
- **Fast exact lookup.** Euclidean top-k is exactly equivalent to

  \[
  xW^\top-\frac12\|W\|^2.
  \]

---

## Notes on the Previous Torus Geometry

The previous Lexicon geometry used the flat torus

\[
T^D=[-1,1)^D=\mathbb{R}^D/2\mathbb{Z}^D
\]

with wrapped-MSE lookup:

\[
d_T^2(x,w_i)
=
\frac{1}{D}
\left\lVert
\mathrm{wrap}(w_i-x)
\right\rVert_2^2.
\]

That geometry is homogeneous, edgeless, and coordinatewise modular. However,
exact lookup requires coordinatewise wrapped displacement, which is less
matmul-friendly than unit-ball L2.

The unit-ball Lexicon is therefore not a drop-in geometric equivalent. It is a
replacement design optimized for faster exact lookup over large vocabularies.

---

## See also

- [`bin/Layers.py`](../bin/Layers.py) — `Lexicon` class.
- [`bin/embed.py`](../bin/embed.py) — embedding training loop.
- [`bin/Spaces.py`](../bin/Spaces.py) — codebook lookup paths.
- [Spaces.md](Spaces.md) — how the Lexicon is used by PerceptualSpace and
  SymbolicSpace.
