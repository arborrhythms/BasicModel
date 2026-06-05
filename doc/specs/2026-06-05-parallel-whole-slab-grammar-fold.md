# Tier-free grammar fold over a bounded STM

**Date:** 2026-06-05
**Status:** Design — boundaries approved; revised after spec review (per-word PS loop
retained for online word learning).
**Supersedes (in part):** the per-word *full re-parse* (`compose` re-fired over the
growing STM snapshot every word); the P/C/S *tier* machinery inside the reducer.

## 1. Motivation

The current grammar engine runs a **serial per-word loop** in which each word fires
the **full grammar fold** over the growing STM snapshot. Three coupled problems:

1. **Cost — the per-word *re-parse*, not the per-word loop.** For each word $p$ the
   host-side `wordSubSpace.compose` runs `max_rounds = N-1` reduction rounds over a
   length-$\approx p$ snapshot, each invoking the tiling DP: $\approx O(N^3)$ per
   sentence, and $\approx 89\%$ of forward wall-clock (measured: per-word `compose`
   $= 2.39\mathrm{s}$ of a $2.67\mathrm{s}$ forward, CPU eager). It runs eager,
   host-side, **never compiled**. The *per-word loop itself* is cheap (PS is
   $\approx 11\%$); the waste is re-folding the whole prefix every word.

2. **The tier conflict.** Each position carries a C/S *tier* state (`position_tier`),
   moved by `lift` (C$\to$S, $+1$) / `lower` (S$\to$C, $-1$), gated by a per-operand
   tier mask. Propagating it (`next_position_tier`) needs the **discrete** parse
   (`src_left`/`action_op`) from the Viterbi backtrace — which blocks both a
   differentiable path and `fullgraph` capture.

3. **Learning.** Per-word hard routing (`routerWireSerial=both`, default) measurably
   **stalls reconstruction** ($\approx 0.25$ plateau vs $\approx 0.11$ and descending
   without it).

The redesign keeps the per-word ingestion (we learn words as we go) but **replaces
the per-word full re-parse with a capacity-driven structured fold**, and deletes the
tier machinery. The fold stays **discrete** ("a structured fold *is* the parse");
the *scoring* that drives it is **soft** (differentiable, straight-through).

## 2. Overview

The engine ingests words **per-word** (online word learning) into a **bounded STM**
(default $8$ — Miller / human working memory). When the STM overflows, a **structured,
tier-free reduce** fires to free a slot (working-memory pressure drives chunking). At
sentence end a final sweep collapses the remaining constituents toward the root(s).
**SS** soft-scores each reduce (gradient path); **CS** commits the discrete
non-interfering tiling (the parse).

```
for each word w in the sentence:            # per-word; learn the word as it arrives
    p   = P.forward(w)                       # percept + online lexicon update
    STM.push(p)                              # bounded; default cap = 8
    while STM.count > cap:                   # working-memory pressure
        scores = SS.forward(STM.slab)        # soft op/placement superposition
        CS.reduce(STM.slab, scores)          # commit one non-interfering tiling level
                                             #   (lift/lower -> SS codebook round-trip)
at sentence end:
    sweep: CS.reduce ... until <= 1 (or a small forest), <= max(order, N-1) levels
STM holds <= cap folded constituents (a root, or a small forest)
```

Default is `cap = 8` (per-word, capacity-driven, $O(N)$). `cap = N` is a variant
(hold all, no capacity pressure, one whole-slab sweep at end, $O(N^2)$) — Section 7.

## 3. Components and boundaries

### 3.1 P engine (per-word, online) — separated

- **Does:** per-word percept analysis and **online word learning** (the lexicon /
  percept codebook updates as each word arrives, so word $p$'s learning is visible to
  word $p+1$). PS stays **per-word** — not batched — precisely because learning is
  online.
- **Interface:** word $\to$ percept vector $\to$ `STM.push`.
- **Change from today:** P currently runs as a *tier* inside the same `compose` loop
  (`attach_unary_ops(tier=...)`). It moves into its own pass, decoupled from the C/S
  reducer.

### 3.2 SS.forward() — soft analysis

- **Does:** the **soft op/placement superposition** over the current STM working set:
  the forward-backward DP marginals (`binary_tiling_soft_dp`) — per-position and
  per-pair posteriors over copy/reduce and over which op. Differentiable; the gradient
  path. Tier-free (single reduction tier $\Rightarrow$ no tier mask).
- **Also:** the per-reduce `lift`/`lower` **codebook round-trip** (C$\to$S quantize,
  S$\to$C dequantize) that those ops now denote as a side effect rather than a tier
  change.

### 3.3 CS.forward() — structured fold (capacity-driven)

- **Does:** owns the bounded STM. On each push, if `count > cap`, commit a **structured
  reduce** — one Viterbi non-interfering tiling (disjoint adjacent pairs) over the
  $\le$ cap working set, applied via `compact_hard`, freeing $\ge 1$ slot. At sentence
  end, sweep the remaining constituents toward root(s). Straight-through: SS's soft
  marginals are the backward surrogate for CS's forward commitment.
- **Interface:** `(STM slab, SS scores) -> reduced STM`.

### 3.4 STM — bounded working memory

- **Does:** holds the folded constituents; fed per-word; reduced under capacity
  pressure. Capacity is a **parameter**, default $8$ ($= \text{nVectors}$).

**Boundary check:** P, SS, CS each answer *what it does / how it is called / what it
depends on* without reading the others' internals. The STM slab and the soft-score
dict are the only cross-unit interfaces.

## 4. Data flow (default, `cap = 8`)

1. **Per-word ingestion.** For each word: `P.forward` (learn) $\to$ `STM.push`.
2. **Capacity-driven reduce.** While `STM.count > cap`: `SS.forward` (soft scores)
   $\to$ `CS.reduce` (commit one non-interfering tiling level; `lift`/`lower` $\to$ SS
   round-trip).
3. **Sentence-end sweep.** Collapse the remaining $\le$ cap constituents toward
   root(s), $\le \max(\text{conceptualOrder}, N-1)$ levels.
4. **Result.** STM holds $\le$ cap folded constituents (the sentence root, or a small
   forest if it did not fully reduce).

## 5. The fold: structure, bound, trigger

- **Structured reduce.** One Viterbi tiling pass selects a *legal* (non-interfering)
  tiling — copy/reduce decisions over disjoint pairs — so reductions in a level fire
  in parallel. One pass $=$ one tree level. SS scores it softly; CS commits it.
- **Trigger.** Capacity pressure during ingestion (`cap = 8`: fire to fit), and a full
  level-sweep at sentence end (and throughout, for the `cap = N` variant).
- **Bound.** Tree depth $\le \max(\text{conceptualOrder}, N-1)$ — `conceptualOrder` as
  a floor; $N-1$ as the worst case (degenerate right-branching tree), since depth is
  unknown ahead and bounded by sentence length. Replaces `max_rounds = x.shape[1] - 1`
  ([Language.py:4226](../../bin/Language.py)) with the floored bound.
- **Cost.** `cap = 8`: the working set is bounded, so each reduce is $O(\text{cap})$
  and there are $O(N)$ of them $\Rightarrow O(N)$ fold work per sentence. `cap = N`:
  $O(N^2)$. Either way, far below today's $\approx O(N^3)$ per-word re-parse.
- **Capacity floor / cap consistency.** `conceptualOrder` $= \log_2(\text{cap})$: three
  halving levels collapse a *balanced* tree of up to $2^3 = 8$ leaves to a root,
  matching the default cap of $8$. Deeper / less-balanced trees extend toward the
  $N-1$ bound.

## 6. Tier elimination

Collapse C and S into a single reduction tier in C; reach S only as a delegated side
effect:

- **Delete** `op_tier_idx`, `op_tier_delta`, `position_tier`, the tier mask
  ([Language.py:5605-5628, 5736-5757, 5853-5868](../../bin/Language.py)).
- **Delete** the per-tier `all_tiers` loop in `compose`
  ([Language.py:4184-4262](../../bin/Language.py)) — one reduction tier.
- **Re-express `lift`/`lower`** as ordinary C-tier ops whose effect is an SS codebook
  round-trip, not a tier delta.
- With one tier the tier mask is trivially all-legal: the soft DP needs no legality
  gate and the discrete tiling needs no per-operand-tier check.

## 7. STM capacity: 8 (default) vs. N — shared path, one conditional

The two regimes **share the per-word PS ingestion and the structured-reduce primitive**
(SS-score $+$ CS-commit). They differ only in *when the reduce fires*:

- `cap = 8` (default): reduce under capacity pressure during ingestion $+$ end sweep —
  a bounded-stack shift-reduce. $O(N)$. Parallels human working memory.
- `cap = N`: no capacity pressure; hold all, one whole-slab level-sweep at sentence
  end. $O(N^2)$. For the case where the compute trade favours it.

The divergence is the reduce **trigger** (a control-flow conditional), not two parsers.
**Default is $8$.** If a single value must be chosen, it is $8$ (cognitive parallel and
cheaper).

## 8. Gradient and causality

- **Gradient — straight-through, cleanly split.** SS produces the soft superposition
  (backward); CS commits the discrete reduce (forward). The existing hardening, but
  housed in SS (score) / CS (fold), fired under capacity pressure instead of as a
  per-word re-parse.
- **Causality.** `forward()`'s contract: within-sentence training is BERT-style
  masked-LM (**bidirectional**); sentence-level AR is delegated to `InterSentenceLayer`.
  Per-word *ingestion* is left-to-right (online learning), but each *reduce* scores
  bidirectionally over the current working set — consistent with masked-LM. No causal
  constraint is violated.

## 9. What gets deleted, moved, kept

| Action | Item | Location |
|---|---|---|
| **Delete** | tier idx/delta, `position_tier`, tier mask | [Language.py:5605-5628, 5736-5757, 5853-5868](../../bin/Language.py) |
| **Delete** | per-tier `all_tiers` compose loop | [Language.py:4184-4262](../../bin/Language.py) |
| **Delete** | `lift`/`lower`-as-tier-delta | [Language.py:5616-5622](../../bin/Language.py) |
| **Delete** | the per-word *full re-parse* fire | [Models.py:6112-6130](../../bin/Models.py) (`wordSubSpace.compose(_snap)` per word) |
| **Change** | `max_rounds = N-1` $\to$ `max(order, N-1)`, fired on the reduce sweep | [Language.py:4226](../../bin/Language.py) |
| **Change** | reduce is capacity-triggered, not per-word | [Models.py `_forward_body_per_word`](../../bin/Models.py) |
| **Move** | soft scoring $\to$ `SS.forward`; discrete fold $\to$ `CS.forward` | from `LanguageLayer.compose` (host-side) |
| **Revert** | the pure-soft `want_hard` edits (fold stays discrete) | [Language.py](../../bin/Language.py) `BinaryStructuredReductionLayer.forward`, `compose` |
| **Keep** | the per-word PS$\to$STM loop and `_stm_shift_and_push` (the per-word push) | [Models.py](../../bin/Models.py), [Spaces.py:10594](../../bin/Spaces.py) |
| **Keep** | STM fold primitives (`_stm_bounded_reduce_step`, `_stm_reduce_to_single_S`) | [Models.py](../../bin/Models.py) — the capacity reduce / end sweep |
| **Keep / relocate** | `learn_relations_from_stm` | fires once per sentence post-sweep; the host-side hoist becomes trivial |

## 10. Reconciliation with in-session work

- **Kept:** the per-word loop and `_stm_shift_and_push` (per-word push) — both tensorized
  this session and both central to per-word ingestion.
- **Kept:** `_stm_reduce_to_single_S` / `_stm_bounded_reduce_step` — they *are* the
  capacity reduce and the end sweep.
- **Reverted:** the pure-soft `want_hard` edits (suggestion 2) — folding entails the
  discrete parse, so there is no tier-free "skip Viterbi"; the win is firing the reduce
  under capacity pressure instead of re-parsing every word.
- **Dissolved:** the `learn_relations_from_stm` graph break — one reduce/sweep per
  sentence means the host-side learning runs once, outside the captured region by
  construction.

## 11. Testing

- **Online learning preserved.** Word-by-word ingestion still updates the lexicon as
  it goes (a word learned at position $p$ is available at $p+1$).
- **Capacity invariant.** With `cap = 8`, the STM never holds $> 8$ constituents after
  a push-and-reduce.
- **Compile gate.** The capacity-reduce path has far fewer host syncs; extend the
  `fullgraph` gate and assert zero graph breaks.
- **Parse correctness.** Tilings are non-interfering (disjoint pairs); depth $\le$ the
  level bound; a known sentence folds to the expected root.
- **Capacity equivalence.** `cap = 8` and `cap = N` produce identical end-states for
  $N \le 8$; for $N > 8$, `cap = 8` stays $\le 8$.
- **Learning.** Reconstruction curve at least matches the no-per-word-routing baseline
  ($\approx 0.11$ and descending), not the stalled hard-routing plateau.

## 12. Phased implementation (one plan, sequenced steps)

1. **Remove the per-word re-parse; add the capacity reduce.** Keep the per-word
   PS$\to$STM loop; delete the per-word `compose` fire; trigger one structured reduce on
   STM overflow, plus the sentence-end sweep.
2. **Tier elimination.** Collapse C/S, re-express `lift`/`lower` as SS round-trip ops,
   delete the tier machinery.
3. **SS/CS split.** Make scoring (SS) vs. fold (CS) explicit at the method boundary.
4. **STM cap parameter.** `cap in {8, N}`, shared path, default $8$; the `cap = N`
   whole-slab sweep as the alternate trigger.
5. **Revert the soft-`want_hard` edits** and re-confirm the `fullgraph` gate + learning.

## 13. Risks and open questions

- **`lift`/`lower` round-trip semantics.** Re-expressing tier moves as codebook
  round-trips must preserve whatever C$\leftrightarrow$S behaviour relative-rule grammars
  (e.g. MM\_grammar) depend on. Needs a faithfulness check against the existing
  relative-rule path.
- **Reduce granularity under pressure.** On overflow, fire the single best-scoring
  reduction (free exactly one slot) or a whole non-interfering level (free several)?
  One slot is minimal and order-faithful; a level is fewer SS passes. To be decided
  with a small A/B.
- **Forest end-states.** Sentences that do not fully reduce leave $> 1$ constituent;
  downstream STM consumers must accept a small forest, not assume a single root.
- **Start symbol / relative rules.** These currently live at the S tier; collapsing
  C/S must keep the start-symbol and `isEqual`/`isPart` semantics intact.
- **Cap-8 long sentences.** Capacity reduces force chunking for $N > 8$; confirm this
  matches the intended human-memory behaviour and does not regress parses that would
  want $> 8$ simultaneous constituents.
