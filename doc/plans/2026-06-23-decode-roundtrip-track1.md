# Decode round-trip (Track 1) — non-destructive prototype-match recovery

> Corrected design + status. Alec's reframing (validated against the code by a
> multi-agent pass, 2026-06-23): a constituent is NOT reconstructed by inverting
> a lossy symmetric fold — the grammar operators are **non-destructive**, so the
> constituent is recovered by **nearest-prototype match against the codebook**,
> realized as the **top-1 of the soft-superposition chooser**. The bare σ/π
> balanced split is genuinely destructive and is correctly *replaced* on the
> decode path (not inverted), staying only as the no-codebook fallback.

## The mechanism

Recovery = `Ops._binary_op_recommend` (`bin/Layers.py:12544`): given the parent,
return an operand pair that are **real rows of the augmented codebook**
`[bottom; W; top]`, selected by order-filtered argmin/argmax + residual match.
It never fabricates an off-manifold vector and stores no partition.

- **lift / VP**: `Ops.liftReverseAll` → `disjunctionReverse` → recommender.
  `VerbLayer.unapply_verb` is the exact verb-conditioned inverse once the verb
  operand is supplied (G5).
- **lower / DET**: `Ops.lowerReverseAll` → `conjunctionReverse` → recommender.
  "a frog" overlaps the stored count-noun "frog" prototype; order-raise + match.

Premise check (validated): the verb forward is a per-dim diagonal atanh-space
rescale, zero-init, touching only a sparse subset — so `VP(NP)` overlaps `NP`
heavily (non-destructive). The order-raise σ-fold is additive (parent contains
its constituents). The bare σ/π `generate` is a partition-blind balanced split
(`L == R`) — destructive, hence replaced not inverted.

## Status

- **G1 — route lift/lower `.what` through the recommender. ✅ DONE + validated**
  (uncommitted). `LiftLayer.reverse` / `LowerLayer.reverse` (`bin/Language.py`)
  now accept `basis=` and route the `.what` split through
  `Ops.liftReverseAll`/`lowerReverseAll` when the basis codebook `W` is present
  (real, distinct constituents), falling back to the balanced split when not.
  The driver already passes `basis=subspace.what` (`Language.py:5828`). Test:
  `test/test_lift_lower_recommender.py` (recommender path ≠ balanced split; the
  no-basis fallback stays `L == R`). **This closes the "the cat" → "t t"
  partition-blind blocker** — distinct operands are recovered by recognition.
- **G2 — does the bare symmetric base need to be invertible? NO. ✅** Recovery is
  prototype-match, not fold-inversion; the balanced split stays as the
  no-codebook fallback. No code change.
- **G3 — heat-biased candidate restriction (`attention_mode`). ○ follow-on**
  (optional; needed at codebook scale, not for a tiny phrase vocab). The
  `reverse` signatures already accept `left_rows`/`right_rows`/`*_priming`;
  lighting it up = widen the driver isinstance guard (`Language.py:5685`) to
  include Lift/Lower + set `<attention>primer</attention>`.
- **G4 — reverse takes a fresh chooser top-k? NOT required.** Reverse is
  deterministic cached-replay of the forward routing; the *operator* (which rule
  + `.where`) is recovered by replay, the *operand* by the recommender (G1). The
  "top-1 of the soft-superposition chooser" is the conceptual unification, not a
  needed code path.
- **G5 — verb inverse. ○ follow-on** (only for phrases with a verb). Populate
  `verb_what` into the reverse dispatch so `VerbLayer.reverse`'s
  `reverse_required_kwargs` gate is satisfied (the verb code is in the cached
  forward routing).

## What full end-to-end text emission still needs

G1 recovers constituents as **idea vectors** (codebook rows). Emitting them as
**words** ("the cat") needs a codebook-index → surface-word realize + a TRAINED
codebook (the words learned). On the current byte/radix-grain configs there is no
word-grain vocab (lexer `raw`), so the final realize step degrades — the SAME
word/byte-grain coupling Phase D (reasoning) hit. A word lexer / trained
phrase-codebook config (`MM_phrase_decode.xml`) is the vehicle to demonstrate the
full round-trip; the recovery mechanism itself (G1) is now done.

## Unification with reasoning (shared code)

Decode-reverse and the reasoning tool-user are **one operation**: soft-propose
then snap onto a real stored prototype (top-1 over a typed store under an overlap
metric). Decode snaps to `subspace.what` rows via `_binary_op_recommend`;
reasoning snaps to stored ideas via `reasoning.quantize`/`query` (`equal`). The
shared abstraction is `snap(proposal, store, metric, candidates) -> prototype`;
`reasoning.quantize`'s "a model codebook is the richer basis" note points at
exactly `subspace.what`. Converging them is the natural consolidation — and it is
the substrate for **ARMA-as-a-reasoner-tool + reasoner-as-next-sentence-predictor**
(the next workstream): the predictor proposes, the codebook/idea store grounds.
