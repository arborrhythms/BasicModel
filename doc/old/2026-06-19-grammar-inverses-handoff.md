# Grammar-operation inverses for reconstruction goals 1 & 2 — handoff

*2026-06-19. Which GrammarOperations need inverses to support the two
reconstruction training stages (`training-stages.md` stages 1–2):*

- **Goal 1** — reconstruct a sentence **keeping the syntax**, predicting only the
  missing **words**.
- **Goal 2** — reconstruct a sentence with **no syntax and no words** retained:
  the parse-tree-deleted, attention-driven decode (deliverable **(C)** of
  [reading-attention.md §4](reading-attention.md)).

> **The one-sentence finding.** The structural **math** inverses are *already
> complete and parse-independent*; what goal 2 lacks is not inverse *arithmetic*
> but an inverse *driver* — today the decode reads *which* category/rule to emit
> from the stored `generate_rules` (the parse tree); goal 2 must read it from the
> **primed symbolic space** (the attention distribution) instead. So goal 2 is a
> **decoupling**, not a re-derivation. **Goal 1 needs no new inverses at all.**

## Goal 1 — keep syntax, fill words: NO new inverses

Because the parse tree is *retained*, `generate_rules` is available and the
existing reverse path runs unchanged. The work for goal 1 is **masking + leaf
fill**, not inverse-building:

- mask lexical leaves in the input;
- run the existing `reverse()` driven by the (kept) `generate_rules`;
- the lexical codebook reverse renders the masked slots:
  `PartSpace.reverse` (`Spaces.py:11668`, text-mode nearest-neighbour) →
  `Embedding.decode_reverse_meta` (`Spaces.py:5007`, nearest-word lookup);
- loss = reconstruction over the masked leaves only.

Every operator goal 1 touches already has a complete reverse (table below). The
verifier confirmed `_reverse_body`'s mathematical inverse is exact for all paths
that currently fire — no build needed for goal 1.

## Goal 2 — parse-tree-deleted: decouple the driver in 6 places

All structural-math inverses (`ConceptualCombine.reverse`,
`LiftLayer.reverse`/`SigmaLayer.generate`,
`LowerLayer.reverse`/`PiLayer.generate`, `_reverse_perceptual`,
`PartSpace.reverse`, `Embedding.decode_reverse_meta`) are **complete and
parse-independent** — goal 2 reuses them unchanged. The build is to make the
*selection* parse-free. Behind a new `parseFree` reverse mode:

1. **`_chart_generate_from_stm()` (`Models.py:7725`) — stub/gate.** It fires
   `SymbolicSpace.generate` to populate `generate_rules`. Under `parseFree`, skip
   it (no parse tree to build); the decode is driven by the primed space instead.
2. **`_chart_compose_at_C` (`Models.py:7668`) — skip.** The forward chart fire on
   the reverse leg is unnecessary once `_chart_generate_from_stm` is stubbed; the
   reverse infrastructure itself is already complete (verifier: "nothing to build
   here — just don't fire the chart").
3. **`SyntacticLayer.reverse` (`Language.py:~7971`) — replace the cursor pop.**
   It currently pops `generate_rules` per cursor to dispatch each grammar-op
   reverse. With no rules to pop, replace the cursor/rule pop with
   **attention-driven category selection** from the primed symbolic space.
4. **`LanguageLayer.reverse_stack` (`Language.py:5650`) + `reverse`
   (`Language.py:5819`) — a `parseFree` variant.** Today it decodes the top
   slot's `.where` to a rule id and unreduces until terminals. Under `parseFree`:
   read the operator identity from the slot's **`.what` codebook** (not the
   `.where` rule lookup) and drive the unreduce **selection** from the attention
   softmax over which categories to emit. The surface contract (unreduce to
   terminals) is unchanged; only the inner driver swaps.
5. **STM reduce — the relative mask (`Models.py:~7807` + the reduce sweep) —
   make it symbol-driven.** The reduce currently infers relative-ness (which STM
   ideas are relative predicates) from the grammar rules. Decouple it to read the
   **STM symbol repertoire** (`symbol_id == predicate_symbol`) instead — the
   explicit requirement in [reading-attention.md §4](reading-attention.md).
6. **`SymbolicSpace.generate` (rule firing) — stub/replace.** Replace the
   signal-router rule firing with: `set_intent` priming + skip-chart + the
   attention-driven decode of (3)/(4).

**The primed-space infrastructure already exists** (this is why this session's
attention work is the substrate for goal 2): `ReadingAttention`
(`Spaces.py`, the learned `.where` producer) and `GlobalAttention` (the typed
addressable attention over input/STM/LTM/codebook) are exactly the "primed
symbolic space" that replaces `generate_rules` as the decode driver. Goal 2 wires
the decode selection (3)/(4) to read that distribution.

## The full map

| operation | forward | reverse status | goal 1 | goal 2 |
|---|---|---|---|---|
| `BasicModel.reverse()` | `Models.py:8007` | complete | entry; uses kept rules | entry; gate `parseFree` |
| `_reverse_body()` | `Models.py:8117` | exact math; rule-driven selection | works as-is | math reused; selection → attention |
| `ConceptualCombine.reverse()` | `Layers.py:1814` | **complete, parse-indep** | reuse | reuse unchanged |
| `LiftLayer.reverse` / `SigmaLayer.generate` | `Language.py:2607` / `Layers.py:2639` | **complete, parse-indep** | reuse | math reused; *which split* from attention |
| `LowerLayer.reverse` / `PiLayer.generate` | `Language.py:2787` / `Layers.py:2811` | **complete, parse-indep** | reuse | math reused; *which split* from attention |
| `_reverse_perceptual()` | `Models.py:8260` | **complete, parse-indep** | reuse | reuse unchanged |
| `PartSpace.reverse()` | `Spaces.py:11668` | **complete, parse-indep** | reuse (leaf fill) | reuse unchanged |
| `Embedding.decode_reverse_meta()` | `Spaces.py:5007` | **complete, parse-indep** | reuse (render words) | reuse unchanged |
| `_chart_generate_from_stm()` | `Models.py:7725` | parse-tree-dependent | required (builds rules) | **stub/gate** |
| `_chart_compose_at_C` | `Models.py:7668` | parse-tree-dependent | required | **skip** |
| `SyntacticLayer.reverse` | `Language.py:~7971` | parse-tree-dependent | required (pops rules) | **replace** w/ primed selection |
| `LanguageLayer.reverse_stack` / `reverse` | `Language.py:5650` / `5819` | parse-tree-dependent | required | **`parseFree` variant** |
| STM reduce relative-mask | `Models.py:~7807` | rule-driven | required | **symbol-driven** |
| `SymbolicSpace.generate` (rule firing) | `Language.py` (SymbolicSubSpace) | parse-tree-dependent | required | **stub/replace** |
| `ReadingAttention` / `GlobalAttention` | `Spaces.py` (this session) | complete | n/a (infra) | **the decode driver** |

## Before building: dialogue through each `forward()` / `reverse()` pair

This map is the static inventory; it is **not** a license to start editing. Before
(and during) the goal-2 work, Alec and I should **walk through each grammar
operation's `forward()` and its `reverse()`/`generate()` as a pair, in
dialogue** — one operation at a time — and agree, per op:

- what the `forward()` actually computes (the exact tensor contract), and whether
  its `reverse()` is a true mathematical inverse or an approximation;
- for the parse-tree-dependent ops, *what* the `reverse()` currently reads from
  `generate_rules` / the chart, and *what* the primed-space (attention) driver
  should feed in its place under `parseFree`;
- the invariants the pair must preserve (round-trip, shape, the EMA/STE
  boundary) and how we'll test each in isolation.

Do this op-by-op (the table below is the checklist of pairs); don't batch-edit
the reverse path from the map alone. The line numbers may have drifted — confirm
each `forward()`/`reverse()` against the live code as we discuss it.

## Build order (goal 2)

1. add the `parseFree` reverse flag/mode + gate `_chart_generate_from_stm` /
   `_chart_compose_at_C` (items 1–2: cheapest, makes the path *run* without a
   chart, even if the selection is still a placeholder);
2. symbol-driven relative mask (item 5: small, self-contained, also benefits the
   forward);
3. the attention-driven selection in `SyntacticLayer.reverse` /
   `LanguageLayer.reverse_stack` (items 3–4: the core — wire the unreduce
   selection to the primed `ReadingAttention`/`GlobalAttention` distribution);
4. retire/stub `SymbolicSpace.generate`'s rule firing under `parseFree` (item 6).

Gate the whole thing (`<parseFree>` or an `ideaDecode` mode) so default reverse
(rule-driven) stays byte-identical. Loss = reconstruct the words from the idea
alone (the existing reconstruction loss with the parse tree deleted) — this is
[training-stages.md](training-stages.md) stage 2.

## Verification note

This map was produced + adversarially verified by a parallel agent sweep
(19 operations mapped, 8 parse-tree-dependent gaps verified against the code).
The line numbers are as the sweep found them and may drift by a few lines;
confirm at build time. The companion doc-consistency pass + legacy flags are in
`2026-06-19-doc-pass-legacy-flags.md`.
