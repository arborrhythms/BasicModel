# Categorical Reparameterization with Gumbel-Softmax
*Eric Jang, Shixiang Gu, Ben Poole — ICLR 2017*

- arXiv: https://arxiv.org/abs/1611.01144
- (parallel result: Maddison, Mnih & Teh 2017, "The Concrete Distribution",
  https://arxiv.org/abs/1611.00712)

## Why this is in `doc/research/`

Direct relevance to the soft-superposition CKY chart. The chart's
per-cell rule mixture saturates to one-hot in the absence of
temperature control: `compat_score` learns large rule-discriminating
logits, the softmax collapses, and we lose soft-superposition in
practice even though the math is "soft." Gumbel-Softmax with annealed
temperature is the standard fix.

## What it does

Replaces a `softmax(logits)` in the loss path with `softmax(logits / τ)`
plus optional Gumbel noise. The temperature τ controls how peaked the
sample is: τ → 0 recovers one-hot (hard); τ → ∞ uniform.

Schedule (canonical): start τ ≈ 1.0–2.0; anneal exponentially to
τ ≈ 0.1–0.5 over training. Early training has gradient flowing
through every category meaningfully; late training commits.

Straight-Through Gumbel-Softmax (ST-GS) variant: argmax in forward,
tempered softmax in backward — keeps the discrete structure where
downstream code expects a hard choice while preserving differentiable
gradient.

## Application here

Two fixes for the chart's saturation:

1. Apply temperature scaling to `cand_score` before softmax:
   `weights = softmax(cand_score / τ)`. Default τ = 1.0 preserves
   current behavior; setting τ > 1.0 (e.g. 2.0) softens the mixture
   so multiple rules contribute meaningfully early.
2. Bound `compat_score` output magnitude (e.g. `tanh × small_scale`)
   so per-rule logit differences cannot grow without bound and
   saturate the softmax independent of τ.

The two are complementary: bounding compat removes the saturating
pressure; τ adds a tunable knob on top.
