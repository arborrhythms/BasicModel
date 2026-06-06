# Dimensional Governance — deep CS hub, wide PS/SS, invertible $\Pi$/$\Sigma$ bridges

Date: 2026-06-05
Status: design (descriptive now; enforced incrementally — see §9)

## Goal

A small set of dimensional invariants that make every space-to-space
handoff a **pure reshape** and make the $\Pi$ (perceptual) and $\Sigma$
(symbolic) operators **square and structurally invertible**. This
replaces the ad-hoc per-space re-dimensioning that relied on a now-retired
percept$\to$concept lift (`sigma_percept`, Stage 1.C) with a single
coordinated content slab.

Glossary: PS = PerceptualSpace, CS = ConceptualSpace, SS = SymbolicSpace,
IS = InputSpace, OS = OutputSpace. $\Pi$ lives at PS, $\Sigma$ at SS.

## 1. Roles and canonical shapes

- **CS is the deep hub**: `[B, N, D]`. `N` = STM depth (the *only* place
  STM size is set — the STM parameter is just CS's `N`); `D` is large.
- **PS and SS are wide**: distributions over a codebook (think KNN
  returns). PS is CS transposed: `[B, D, N]`.
- **IS and OS are 2-D** (unanalyzed / unsynthesized): `[B, N, 1]` when they
  must be materialized.
- **CS output dim = the common input size of PS, SS, and OS.**

## 2. The square (flat-slab) invariant

Define **content** $= \mathit{nDim} - \mathit{nWhere} - \mathit{nWhen}$
(the `.where`/`.when` band is per-position overhead, not reshaped content).

The content slab $\mathit{nOutput} \times \mathit{content}$ is **constant**
across IS, PS, CS (and the OS handoff). So IS $\to$ PS $\to$ CS $\to$ SS
are pure reshapes — *wide* $\leftrightarrow$ *deep* regroupings of the same
elements, never re-dimensionings.

Because the slab is constant, $\Pi$ and $\Sigma$ are **square** over the
flattened slab and therefore structurally invertible (LDU). **The
operators do not absorb dimension mismatches** — a mismatch is a config
error, not something a layer silently fixes. (This generalizes the
existing `validate_config` flat-slab check, restated in content terms.)

Worked example (MM_5M, band $= 4$): IS `1024 x 8 = 8192`,
PS `1024 x 8 = 8192`, CS `8 x 1024 = 8192` — the wide PS regroups into the
deep CS with no loss.

## 3. The $\Pi$ / $\Sigma$ bridges

$\Pi$ at PS = **subsymbolic analysis**; $\Sigma$ at SS = **symbolic
synthesis**. Symmetric around the deep CS hub:

```
        PI (invertible)            SIGMA (invertible)
   PS  <----->  CS (deep, D)  <----->  SS
 (wide percepts)                  (wide symbols)
```

- `CS @ PI -> distribution over PS percepts`;
  `percept_dist @ PI_inverse -> CS`.
- `CS @ SIGMA -> distribution over SS symbols`;
  `symbol_dist @ SIGMA_inverse -> CS`.

**Key property:** the deep `D`-vector lives in the *bridge matrix*, not in
any single percept or symbol entry. No individual entry carries the full
idea; only the whole distribution back through the inverse reconstitutes
it. The codebook is the bridge matrix plus lightweight entries, not a
table of `D`-vectors (which would blow the parameter budget — e.g.
$131072 \times 1028 \approx 135\text{M}$).

**`<sigmaPi>last|butterfly|full</sigmaPi>`** (replaces `<butterfly>`):
the $\Pi$/$\Sigma$ matrices operate on

- `last` — the last dim only (per-slot square; current default, back-compat),
- `butterfly` — an $O(N \log N)$ cross-dim cascade (= today's
  `<butterfly>true</butterfly>`),
- `full` — the full flattened square matrix (full interconnect; this is
  the square bridge of §2).

Default `last`. Applies to PS.$\Pi$ and SS.$\Sigma$.

## 4. PS / SS asymmetry (the only one)

Everything above is symmetric. The lone asymmetry is the selection +
transport on top of the bridge:

- **PS = choose-1, continuous.** The percept distribution collapses to the
  nearest single percept, transmitted as an unquantized continuous code;
  `PI_inverse` reconstructs CS.
- **SS = choose-N, quantized / by-reference.** The symbol distribution
  keeps its top-N, transmitted quantized. Each symbol is a small entry
  (content width, e.g. a 4-dim code); the CS idea is recovered via
  $\Sigma$ + the **CS codebook quantization** (the symbol references a CS
  concept; the small SS code disambiguates symbols that share a concept and
  can carry a learned, in-band part-of-speech / role signal — matching
  role-collapsed's "POS is learned from participation").

The pairing is realized by the **$\Sigma$ matrix** (Option A of the design
dialogue), not by storing a private `D`-vector per symbol: `SS_code @
SIGMA -> CS idea`, then CS snaps on its own codebook. This keeps the
$\Pi$@PS / $\Sigma$@SS symmetry intact and is compatible with
`architecture.monotonic` (a CS `project` codebook is not).

## 5. Directionality

- **SS:** `CS -> symbols` is the *forward* synthesis.
- **PS:** `CS -> percepts` (`PS.forward(CS)`) is the *top-down /
  reconstruction* leg; the bottom-up `IS -> percepts -> CS` leg is the same
  $\Pi$ inverted (`percepts @ PI_inverse -> CS`). One invertible matrix
  serves both legs — which is why the scheme stays clean.

## 6. Parallel vs serial

- **Parallel** (MM_5M): $\Pi$ and $\Sigma$ are the literal matmul bridges
  over the wide $\leftrightarrow$ deep reshape.
- **Serial** (MM_5M_grammar): **grammatical analysis** replaces the PiLayer
  (on the input / incoming CS); **grammatical synthesis** = the
  bounded-STM **fold** (from the 2026-06-05 tier-free refactor) replaces
  the SigmaLayer. One word at a time; CS supplies 1-2 ideas as
  mask/steering; a single (quantized) idea returns. A serial symbol's
  content is `[activation + where(2) + when(2)]` of the event width: with
  the $\Sigma$-bridge pairing the activation is the 4-dim code
  ($4+2+2 = 8$); a bare scalar activation would be $1+2+2 = 5$.

## 7. Reference configurations (built as part of this work)

- **`data/MM_5M.xml`** — parallel; $\approx 5$M parameters (dominated by
  the SS symbol codebook); CS deep (`N=8`, large `D`), PS/SS wide,
  `<sigmaPi>full</sigmaPi>`, all per this spec.
- **`data/MM_5M_grammar.xml`** — serial; **matching MM_5M's shape**; param
  count dictated by the serial grammar-op budget (small ops). Uses
  `role_collapsed.grammar`.

## 8. Grammar finalization

- Convert `xor`, `default`, `shamatha` grammars to the **role-collapsed**
  format (role-based `op_I[n]` / `op_O1`, `<start>` declarations). Keep
  `complete.grammar` as the retained compatibility baseline.
  `role_collapsed.grammar` is the default.

## 9. Implementation phasing (the agreed sequence)

1. **Minimal working paradigm** — code + configs to get MM_5M (parallel
   $\Pi$/$\Sigma$ bridge + `<sigmaPi>` modes) and MM_5M_grammar (serial
   fold + grammar conversion) forward-passing and reconstructing under the
   new rules. Config alone is *not* sufficient: the wide$\leftrightarrow$deep
   bridge is the core new code.
2. **Tests** — flat-slab + invertibility + capacity gates for MM_5M;
   serial fold + grammar gates for MM_5M_grammar; both configs build,
   forward, and reconstruct.
3. **Reduce flexibility (test-first)** — only after 1-2 pass: enforce the
   flat-slab invariant globally (operators no longer absorb dim
   mismatches), migrate existing configs to comply, and remove the legacy
   re-dimensioning paths (the retired-lift machinery).

## 10. Non-goals / deferred

- The concrete **by-reference SS transport** encoding (how the top-N
  symbol references are serialized) — the spec only requires the
  $\Sigma$-bridge + codebook-reference *semantics*.
- **Global enforcement** is Phase 3, deliberately not Phase 1 — existing
  files keep their flexibility until the paradigm is proven on the two
  reference configs.

## Self-review notes

- No placeholders/TBDs except the explicitly-deferred §10 items.
- §4 (activation width 1 vs 4) is resolved to the 4-dim $\Sigma$-bridge
  code; the 1-dim scalar is noted only as the bare-activation alternative.
- Scope: one paradigm + two reference configs + grammar conversion; global
  enforcement is split out to Phase 3 to keep this implementable as one
  plan.
