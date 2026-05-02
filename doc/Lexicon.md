# Lexicon

The Lexicon's narrative description and distance math have moved into
[Spaces.md — Lexicon (Projective Unit Ball)](Spaces.md#lexicon-projective-unit-ball),
co-located with the Codebook Similarity Metric and the per-space
geometry discussion (PerceptualSpace, ConceptualSpace, SymbolicSpace).

## Quick reference

The Lexicon ([`bin/Layers.py`](../bin/Layers.py)) is a learnable
vocabulary embedding on the **projective unit ball** $B^D / (x \sim
-x) \cong \mathbb{RP}^D$ — the closed unit ball with the
**negation identification** $w \sim -w$ (the $\pm$-quotient). Note
that $-w$ is the *negation* of $w$, not its **antipode**: the antipode
of a point is the furthest point in the manifold (used by SBOW as a
repulsion target), and on $\mathbb{RP}^D$ it is the orthogonal
hyperplane, not a unique point. Distance and lookup:

$$
d_{\mathbb{RP}}^2(a, b) \;=\; \min(\|a-b\|_2^2,\ \|a+b\|_2^2)
\;=\; \|a\|_2^2 + \|b\|_2^2 - 2\,|\langle a, b\rangle|,
$$

$$
\operatorname{score}(x, w_i) \;=\; |\langle x, w_i\rangle| - \tfrac{1}{2}\|w_i\|_2^2.
$$

Top-k lookup is one matmul, one elementwise abs, one broadcast subtract.

```python
lexicon = Lexicon(V, D)              # default: projective unit ball
lexicon.project_unit_ball_()         # call after optimizer.step()
W_index, W_norm2 = lexicon.lookup_index()
idx, dist_sq, scores = Lexicon.topk_rp(x, W_index, W_norm2, k=32)
```

For the full derivation, SBOW pode/antipode gradient consequences,
chunked-lookup helpers, and the legacy torus primitives kept for
backward compatibility, see
[Spaces.md — Lexicon (Projective Unit Ball)](Spaces.md#lexicon-projective-unit-ball).

## See also

- [`bin/Layers.py`](../bin/Layers.py) — `Lexicon` class and `topk_rp` /
  `topk_rp_chunked` helpers.
- [`bin/embed.py`](../bin/embed.py) — SBOW training loop.
- [Spaces.md](Spaces.md) — full per-space geometry discussion, including
  the contrast between the projective Lexicon (PerceptualSpace,
  SymbolicSpace) and ConceptualSpace's unit-direction codebook.
- [test/bench_codebook_lookup.py](../test/bench_codebook_lookup.py) —
  performance comparison of the broadcast, matmul, pole-aligned, and
  chunked-wrap forms.
