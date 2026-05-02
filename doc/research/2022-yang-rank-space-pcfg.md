# Dynamic Programming in Rank Space: Scaling Structured Inference with Low-Rank HMMs and PCFGs
*Songlin Yang, Wei Liu, Kewei Tu — NAACL 2022*

- ACL Anthology: https://aclanthology.org/2022.naacl-main.353/

## Why this is in `doc/research/`

If we ever stratify the soft-superposition chart's grammar into
genuine multi-category PCFG form (S, NP, VP, AP, MP, PP, ... — see
`doc/old/Grammar.md`), the chart's R × C × C scoring tensor at every
cell becomes the bottleneck. CP/Tucker decomposition on the rule
score tensor lets the chart scale to grammars with hundreds of
categories without quadratic blowup.

## What it does

Standard inside-pass cost is O(N³ · C³ · R) for `Σ_{r, k} P(r) ·
inside[i, k, B_r] · inside[k, j, C_r]`. Yang et al. apply CP
decomposition (rank R') to the rule-probability tensor and rewrite
the inside recursion in **rank space**: per cell, marginalize over
rank-r' instead of (rule, B, C). Cost drops to O(N³ · R' · C) when
R' < C².

For our use case (small grammars, R ≤ 30), this isn't yet a hot
issue, but the technique is on the shelf when the grammar grows. Note
also that this paper's framing — rank-space DP — naturally aligns
with bivector / lift-lower geometry already present in the project.

## Earlier related: Cohen 2013

"Approximate PCFG Parsing Using Tensor Decomposition" —
https://aclanthology.org/N13-1052.pdf — first to apply tensor
decomposition to PCFGs. Yang et al.'s 2022 paper is the modern
neural-friendly version.
