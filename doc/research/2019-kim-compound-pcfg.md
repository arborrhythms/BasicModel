# Compound Probabilistic Context-Free Grammars for Grammar Induction
*Yoon Kim, Chris Dyer, Alexander Rush — ACL 2019*

- arXiv: https://arxiv.org/abs/1906.10225
- PDF: https://arxiv.org/pdf/1906.10225
- Anthology: https://aclanthology.org/P19-1228/
- Reference implementation: https://github.com/harvardnlp/compound-pcfg

## Why this is in `doc/research/`

The closest published peer to the floating-blossom soft-superposition CKY chart
in [`bin/Language.py`](../../bin/Language.py)
(`SyntacticLayer._compose_chart_cky`). Read this before extending or refactoring
that path.

## What the paper does

Unsupervised grammar induction with a probabilistic CFG whose **rule
probabilities** are produced by a neural network conditioned on a per-sentence
latent `z`:

```
P(rule | z) = softmax(MLP_rule(z))
```

Inference uses the **standard inside algorithm** (logsumexp over splits and
rules per chart cell). With `z` known the model collapses to a deterministic
PCFG, so latent trees are marginalized exactly via dynamic programming;
`z` is handled with amortized variational inference (reparameterization).

## The architectural choice that matters here

> **Rule semantics are fixed. Only rule probabilities are learned.**

Each production `A -> B C` is a symbolic concat with the `[A]` label; the
neural network does not learn a per-rule MLP that interpolates the children
into a parent vector. The chart's only learned signal flows through `P(rule)`.

The corollary, in our setting:
- **Compose must dispatch to fixed rule semantics**, e.g. `intersectionForward`,
  `unionForward`, `notForward`, `liftForward`, `lowerForward` — the same ops
  the legacy path uses.
- The chart's **soft superposition is over rule probabilities only**, not over
  a learned `Compose` MLP.
- Hardening at test time (Viterbi argmax over rules per cell) is meaningful
  because `P(rule)` was the only adjustable component during training, so the
  argmax encodes what the model believes the grammar is.

## How the floating-blossom chart deviated (and why we're refactoring)

The original Delta 2 introduced `_SoftCompose(left, right, rule_embed,
marker_mask) -> vec` — a learned MLP that produces the parent vector from the
operands plus a rule embedding. That gave the chart a learnable interpolant
between operands and parent, which absorbs whatever function the loss demands.
On XOR, the chart's hardened grammar collapsed to a single rule
(`intersection`) while downstream layers carried the actual XOR computation —
the chart's rule selection was underdetermined because the MLP could fit any
function regardless of which rule fired.

The refactor (in progress) replaces `_SoftCompose` with a per-rule dispatch
through `_RULE_METHODS`, matching Kim et al.'s fixed-semantics constraint.
`_rule_bias`, `_rule_embed`, and `_compat_score` remain as the learned scoring
path; `_SoftCompose` is removed from the vector path.

## Project-specific structural concern (post-refactor open question)

`NegationLayer` lives only in `SymbolSpace` in this codebase, on the
hypothesis that lower spaces represent a "mind without negation in harmony
with the world." A grammar of the canonical XNF shape

```
S = not(or(and(X)))
```

implies a depth-graded composition where `and` operates near the leaves, `or`
mid-level, and `not` only at the top. The chart should respect this layering;
allowing `not` at any depth would let the chart bypass the structural
constraint that makes the lower spaces additive-only.

Two open design choices follow:
1. Whether the chart's per-rule `marker_mask` (or a similar gating mechanism)
   should encode depth/space-space-role eligibility, blocking `not` from firing
   below the `S` cell.
2. Whether two orders of composition (one for `and`+`or` at the inner spaces,
   one for `not` at the symbolic boundary) need separate chart passes, or
   whether a single chart with depth-conditioned rule masks suffices.
