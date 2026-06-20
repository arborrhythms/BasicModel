# Reading attention, global attention, and idea decoding

*Spec authored 2026-06-19. Roadmap for the remaining attention work.*

> **Status: design / roadmap.** Everything here is **gated and dark by
> default** — byte-identical when off. It builds on the substrate already in the
> tree (the priming machinery, the `<mereologyRaise>` top-down handoff, the
> relation/symbol tables, the reverse/generate path) and on the design in
> [orders.md §6](orders.md) "The attention substrate". This document is the
> *build plan*: what exists, what to build, and the three deliverables.

The three deliverables:

- **(A) Reading attention** — a learned `.where` producer behind
  `<readingAttention>` that replaces the serial reading for-loop. It is the
  learned producer of the `_passback_scope_where` the `<mereologyRaise>` handoff
  already consumes.
- **(B) Global attention** — free, content/relation-driven attention. It
  **needs a stochastic element first** (the existing selection is deterministic);
  reading is introduced before global.
- **(C) Idea decoding** — decode a C-tier idea into words when no forward
  grammar/parse is present, by building the activated attention space, **deleting
  the parse tree**, and driving the grammar `.reverse()` methods from just the
  primed symbolic space.

---

## 1. The substrate that already exists

Reading attention is a *producer* for, and global/decode are *consumers* of, a
substrate that is largely built but unwired into a learned attention. What's
there today:

| piece | where | role |
|---|---|---|
| `set_intent` / `intent_boosts` / `install_intent_priming` / `selection_boost_fn` | `Spaces.py:7090-7150` | priming: a `[V]` boost from one intent code against the codebook rows, multiplied into VQ row selection. The codebook-retrieval prior. |
| `intent_priming_weights` | `Spaces.py:102-133` | computes the `[V]` boost (graded similarity of intent vs tower rows). |
| `<attention>` modes (`off`/`primer`/`second-order`/`low-rank`) | `Spaces.py:182`, consumed in `Language.py` | taxonomy-heat **retrieval** (spreading activation over the lattice) — NOT QKV. |
| `RunStructureLayer` / `route_hint` | `Layers.py:3112-3229` | run/gap/containment over `.where` brackets; `route_hint ∈ {0 NULL, 1 REFINE, 2 RAISE}`. |
| `WholeSpace.passback_action` | `Spaces.py:18610` | the 4-case WS→PS dispatch (noop/scoped/refine/chunk); reads `_passback_scope_where`, `intent_boosts`, `_mereology_ratio_obs`. |
| `BasicModel._passback_scope_ps` / `_passback_scope_where` | `Models.py:7807`, `Spaces.py:18645` | the consumer of the reading `.where`; scoped case focuses the percept to a `[start,end]` bracket. |
| `create_word_object_meta` | `Spaces.py:13103` | mints A=word / B=object / C=reify-meta symbols (relation-only symbol table). |
| `RelativeTruthStore` / `record_triple` / `_maybe_learn_relation` / `learn_relations_from_stm` | `Layers.py:7305`, `Spaces.py:13901-14015` | relations between ideas (uncollapsed triples), learned at the sentence boundary. |
| `SubSpace._index` / `.active` | `Spaces.py:4829-4859` | the per-position index/activation — the discrete channel the symbol table rides on. |
| serial reading loop `_forward_body_per_word` / `_per_word_body_step` | `Models.py:7214`, `7016` | the deterministic per-word cursor reading attention replaces. |

**The one missing piece**: there is *no producer* of `_passback_scope_where`.
`passback_action` only **reads** it (the scoped case); nothing in production
**sets** it yet (only the tests do). Reading attention is that producer — the
hook is already in place, waiting for it.

---

## 2. (A) Reading attention — the learned `.where` producer (`<readingAttention>`)

**Goal.** Replace the serial reading for-loop with a learned `.where` attention
that, at each `t>0` subsymbolic pass, emits the next reading scope into
`_passback_scope_where`; the existing `<mereologyRaise>` handoff
(`_passback_scope_ps`) consumes it.

**Mechanism (cross-attention = spreading activation, not QKV).**

- **query** = the post-σ/π concept `CS_t` *plus* the symbols currently active in
  STM (so expectation flows top-down).
- **keys** = the **surface analysis spans** (`_staged_analysis_spans`, the fixed
  word `.where` brackets) *and* the **CS-owned symbol table** (indices into the
  separate PS/WS codebooks — see orders.md §6).
- **score** = a fused sum of a **subsymbolic** term (codebook-retrieval prior:
  the existing `intent_boosts` / `selection_boost_fn` path) and a **symbolic**
  term (relation/co-occurrence over the symbol+relation tables). One distribution
  over positions → `next_where` (soft `Σ αᵢ·spanᵢ`, hard argmax at inference).
- `next_where` → `_passback_scope_where` → the handoff scopes PS/WS.

**Loss.** Text mode only, gated: cross-entropy of the attention distribution to
the **next word's index** ("penalize attention that does not select the next
word"). CE over discrete spans gets the `.where` *extent* for free (each candidate
carries its own width).

**Trainability discipline.**

- **Monotonic / coverage** — mask already-consumed spans (or a monotonic-energy
  bias) so reading stays left-to-right and can't re-select.
- **Teacher forcing** — feed the true `next_where` as the scope during training
  while the loss trains the producer to predict it; free-run at inference.
- **Shift bootstrap** — initialize the advance as "shift the bracket forward by
  ~its own extent" (a phase rotation in the endpoint-sum quadrature). At init it
  *is* the for-loop; the loss only refines it. Degrades gracefully.

**Gradient discipline.** The loss input is the **primed symbols**; gradient
**stops there** — it does NOT backprop into the codebooks, so the EMA-only /
detached-rows VQ contract (C-9/C-11) is intact. A symbol's importance reaches its
codebook rows indirectly: attending it activates its referenced rows → those rows
occur → EMA consumes the occurrence (free, given indices-not-vectors). Relations
may take error directly.

**Gating.** `<readingAttention>` (default off → byte-identical). The producer
writes `_passback_scope_where` in the per-stage loop / `_per_word_body_step`
before the handoff hook; the serial reading loop becomes the *supervised target*
the producer is trained against (then retired once the attention tracks it).

---

## 3. (B) Global attention and the stochastic element it needs

**Local vs global.** Reading attention (§2) is *local*: next-word, monotonic,
**supervised** (the next-word target breaks symmetry). **Global** attention is
free — content/relation-driven, able to land anywhere, including the abstract
relations-of-relations that have *no mereological `.where`*. It has **no direct
supervised target**, so it cannot break symmetry or discover what to attend by
itself: **it needs a stochastic element to explore first.**

**Why stochastic (grounded).** Today the selection is deterministic — VQ is hard
`argmax` (`Spaces.py:2814,6335`), `_topk_priming_mask` is deterministic top-k
(`Spaces.py:18554`). The workflow gap is explicit: there is *no* temperature-
scaled softmax over the codebook, *no* soft blending across entries, *no* global
marginalizable attention distribution. A free attention that explores needs a
differentiable, samplable selection so that the downstream task error (not a
next-word target) can shape it.

**The stochastic primitive to reuse.** It already exists for the grammar chooser
and just needs extending to the codebook/attention selection:

- **two-pass soft-superposition `<learning>`** (`Models.py:777-789`,
  `Language.py:7275`): pass A at superposition temperature 0 (sharp, recorded),
  pass B at `exploreTemperature` (flat, exploration, trimmed from the batch). The
  chooser sits **directly in the gradient path** (`superposition_scale(t)=1-t`,
  `Language.py:5875`). This is exactly stochastic exploration over a selection.
- (alternative) **ergodic noise** `W_eff = bias·W + var·noise`
  (`Layers.py:927,937,...`, `resample_noise`, `_ergodic_var`).

**Plan.** Introduce **reading first** (local, supervised). Introduce **global
attention only once the stochastic element is present**: global attention reuses
the §2 machinery but adds a **temperature-superposition over the selection** (the
two-pass scheme extended from the chooser to the codebook/attention distribution),
trained by the downstream task error rather than the next-word loss. Gate
`<globalAttention>` so it *requires* the stochastic element (e.g. `<learning>` /
`exploreTemperature` on); off → no global attention.

**Build (✗):** a temperature-scaled softmax / soft-superposition over codebook
rows and over the position distribution (the missing primitive), wired to the
two-pass `<learning>` exploration so global attention is samplable + gradient-
shaped.

---

## 4. (C) Decoding an idea into words (the parse-tree-deleted reverse)

**The problem.** Decode a C-tier idea (in STM) into words when there is **no
forward grammar/parse** — i.e. *generation/recall*, where there is nothing to
invert.

**The path that exists** (grounded — it already generates a sentence for a
top-down idea):

```
BasicModel.reverse()                         Models.py:8007
  → _chart_generate_from_stm()               Models.py:7660  (fires SymbolicSpace.generate,
                                                              populates generate_rules)
  → _reverse_body()                          Models.py:7860  (per-stage inversion;
                                                              ConceptualCombine.reverse + grammar reverse)
      · LiftLayer.reverse / SigmaLayer.generate   Language.py:2607, Layers.py:3039 (balanced split)
      · LowerLayer.reverse / PiLayer.generate     Language.py:2787, Layers.py:4088 (log-mult split)
      · LanguageLayer.reverse_stack / generate    Language.py:5650, 4876 (unreduce to terminals)
      · SyntacticLayer.reverse                     Language.py:5819 (pops generate_rules per cursor)
  → _reverse_perceptual()                    Models.py:8003
  → InputSpace/PartSpace.reverse()           Spaces.py:8583, 11254
  → Embedding.decode_reverse_meta()          Spaces.py:4567  (NEAREST-WORD codebook reverse)
```

**The dependency to break.** This decode is driven by `generate_rules` — the
stored rule sequence `_chart_generate_from_stm` reads off the STM idea. That
sequence **is the parse tree** (a chart record). The user's scheme decodes
*without* it.

**The new scheme (the three steps).**

1. **Build up the activated space of attention.** The idea primes the symbolic
   space — spreading activation over the CS symbol table + relations
   (`set_intent` from the idea, the relation-store score) — producing an
   activated distribution over symbols/concepts: the *generative prior* for what
   this idea is about.
2. **Delete the parse tree.** Clear the chart / `generate_rules` / the STM
   reduction structure — grounded, this is the sentence-boundary reset:
   `stm.clear`, the CS `event` drop, `SymbolicSpace` soft-reset,
   `LanguageLayer` rule caches. Keep **only** the idea + the primed attention
   space. The parse tree is transient (rule firings in
   `symbolicSpace.current_rules`); the persistent state is just STM content+depth.
3. **Decode from the primed symbolic space via grammar `.reverse()`.** With no
   stored rule sequence, the grammar `.reverse()`/`.generate()` operations
   (`Lift`/`Lower`/`Sigma`/`Pi` balanced splits, `reverse_stack`) are driven by
   the **primed symbolic space** to choose which categories/words to emit at each
   split. Attention **replaces `generate_rules`** as the decode driver; the
   nearest-word codebook reverse renders the chosen symbols to words.

**Training ("building up the activated space of attention" *is* the training).**
An autoencoder over the parse-tree bottleneck:

```
forward-parse a sentence (teacher) → idea in STM (+ the parse tree)
   → DELETE the parse tree
   → regenerate the words from JUST the idea-primed attention space
      via grammar .reverse()
   → loss = reconstructed words vs original   (the existing IR / reconstruction loss)
```

The loss forces the primed attention space to carry enough generative structure
to reproduce the words *without* the parse tree. The same primed substrate that
drives reading (encode, §2) drives decoding (generate).

**Build (✗) — the decouplings the workflow surfaced:**

- decouple the STM reduce sweep from the CKY `_chart_compose_at_C` (today the
  reduce reads `current_rules` for the relative mask);
- make the **relative-mask symbol-driven** — infer relative-ness from the STM
  symbol repertoire (e.g. `symbol_id == predicate_symbol`) instead of querying
  grammar rules;
- a `parseFree` reverse path that skips/stubs `compose-at-C` and drives the
  grammar `.reverse()` selection from the primed symbolic space;
- pre-assign / read symbol IDs on the STM (alongside depth/content) so "primed
  symbolic space" is well-defined without a chart.

---

## 5. How the three tie together

One substrate, three uses:

- **reading** (encode) — supervised, local; *builds* the substrate and is the
  `.where` producer (§2).
- **global** (free) — stochastic; *extends* the substrate to abstractions with no
  `.where` (§3). Requires the stochastic element.
- **decoding** (generate) — *consumes* the primed substrate to drive the grammar
  `.reverse()` without a parse tree (§4).

All three pump the same orders (orders.md §6): `subsymbolicOrder` over the
mereological substrate, `symbolicOrder` over the relational substrate (the pump
that surfaces the relations-of-relations decoding needs for ideas-of-ideas),
`syntacticOrder` the parse-tree depth the relational pump may compose to.

---

## 6. Flags (all default off → byte-identical)

| flag | gates |
|---|---|
| `<readingAttention>` | **(LANDED, §9)** the learned `.where` producer (§2); writes `_passback_scope_where` for the existing handoff + adds the text-mode next-word CE loss. |
| `<learning>` / `exploreTemperature` (exists) extended to selection | **(LANDED, §10)** the stochastic element (§3): the two-pass superposition temperature now scales the attention selection (`ReadingAttention.superposition_scale`) — both reading and global. |
| `<globalAttention>` | **(LANDED, §10)** free attention over the typed addressable space (§3); requires the stochastic element. |
| `<ideaDecode>` (or a `parseFree` reverse mode) | the parse-tree-deleted, attention-driven decode (§4). |

---

## 7. Remaining-work checklist

**✓ exists:** the priming machinery (`set_intent`/`intent_boosts`/
`selection_boost_fn`); the `<attention>` retrieval modes; `RunStructureLayer`/
`route_hint`; the `<mereologyRaise>` handoff + `_passback_scope_where` hook; the
relation/symbol tables (`create_word_object_meta`/`RelativeTruthStore`/
`_maybe_learn_relation`/`learn_relations_from_stm`); the full reverse/generate
path (`reverse`/`_chart_generate_from_stm`/`_reverse_body`/grammar `.reverse()`/
`Embedding.decode_reverse_meta`); the stochastic two-pass (`exploreTemperature`/
soft-superposition) + ergodic noise.

**✓ build (A) — LANDED 2026-06-19 (gated `<readingAttention>`, dark by
default):** the learned `.where` producer (`Spaces.ReadingAttention`) writing
`_passback_scope_where`, with the next-word CE loss, monotonic/coverage mask,
teacher forcing, and shift bootstrap (§2). See "Implementation (A)" below.

**✓ build (B) — LANDED 2026-06-19 (gated, dark by default):** the stochastic
element (the two-pass superposition temperature on the attention selection) +
global attention over a typed addressable space {input window, STM, LTM, symbol/
whole codebook}. See "Implementation (B)" below.

**✗ build:**

3. the parse-tree-deleted decode: prime from idea → delete the parse tree →
   grammar `.reverse()` driven by the primed space → reconstruction loss; plus
   the decouplings (STM↔CKY, symbol-driven relative mask, `parseFree` reverse)
   (§4).
4. **global-attention consumer + book paging** (deferred, Alec): feed the
   soft-read content back into the forward + train by the downstream task error;
   and the out-of-core hierarchical `.where` over a pre-chunked book index (the
   coarse-index representation — codebook-as-index vs separate RAG index — is
   "decide later").

---

## 9. Implementation (A) — the reading-attention producer (LANDED)

`Spaces.ReadingAttention` (an `nn.Module`) is the learned `.where` producer; it
is built on `BasicModel` (`self.reading_attention`) when `<readingAttention>`
is on, and `None` otherwise (no module, no scope, no loss — byte-identical).
`BasicModel._reading_attention_step` runs it at each `t>0` subsymbolic pass in
the parallel per-stage body (`_forward_body`), **before** the `<mereologyRaise>`
handoff (`_passback_scope_ps`) so the "scoped" branch consumes the
freshly-produced scope.

- **query** — the prior pass's concept (`prevCS_forSS`, the post-σ/π CS→WS
  view, mean-pooled) as the **subsymbolic / mereological** retrieval term, plus
  the active STM symbols (`cs.stm.snapshot`, mean-pooled) as the **symbolic**
  term. Both DETACHED.
- **keys** — the staged analysis spans (`_staged_analysis_spans` `[B,K,2]`, the
  whitespace word brackets) pooled to per-span percept content (DETACHED).
- **score** — a small MLP over six DETACHED features: the **subsymbolic
  codebook-retrieval prior** (the literal `intent_boosts` path:
  `maxᵥ( cos(span, rowᵥ)·boostᵥ )` over the percept/PartSpace codebook, the
  `intent_priming_weights` + `(sim·boosts).amax` reduction
  `WholeSpace._topk_priming_mask` uses; `boostᵥ` is the tower's primed-intent
  state if set, else derived from the concept; **concept-content cosine
  fallback** when the space carries no codebook), the **symbolic** term
  `cos(symbols, span)`, the span geometry (`start/N`, `end/N`, `extent/N`), and
  the signed cursor distance. Plus a **non-learned shift-bootstrap bias** that
  favours the next unconsumed span;
  the readout head is **zero-init**, so at init the logits ARE that bias — the
  producer is the serial reading for-loop (`argmax = read_idx = t−1`), and the
  CE loss only refines it.
- **mask** — consumed spans (`k < read_idx`) and padding spans (extent 0) are
  set to `−∞` (monotonic / coverage). The distribution is a softmax over the
  masked logits; `next_where = Σ αₖ·spanₖ` (normalized `[start, end]`).
- **scope write** — `wholeSpaces[0]._passback_scope_where`: the TRUE next span
  during training (teacher forcing, so PartSpace reads the right word while the
  producer learns to predict it) and the predicted soft span at inference
  (free-run); collapsed to one `[start, end]` (mean over rows with a real next
  word) for the batch-collapsing handoff. `None` once the cursor runs past the
  last word.
- **loss** — text mode only (`model_type == "embedding"`, training): the
  next-word CE (`−log α` at `read_idx`) added to `CS_sub.errors`
  (`reading_attention`, category `symbol`), which `copy_context` shares through
  the pipeline into the model's `totalLoss`. The readout params are added to the
  optimizer in `getOptimizer` (the `self.spaces` walk misses a model-level
  module).
- **gradient boundary** — every score input is detached, so the loss trains the
  MLP readout ONLY; it never backprops into the EMA-only VQ codebooks
  (C-9/C-11) nor the primed symbols (orders.md §6 "Learning").

**Config:** `data/MM_reading.xml` (derived from `MM_mereology`; adds
`<readingAttention>true</readingAttention>` + WholeSpace `<analysis>word`).
**Tests:** `test/test_reading_attention.py` (the module in isolation — incl. the
codebook-retrieval prior + its gradient boundary — and the model wiring).
**Remaining refinement (hook, not deviation):** the **symbolic** term rides the
STM-symbol query rather than materializing the CS symbol table as separate keys
(a relation/co-occurrence index); the table-as-keys + relation-store bias is the
natural extension toward (B) global attention. *(The subsymbolic term now uses
the literal `intent_boosts` codebook-retrieval prior — see "score" above.)*

---

## 10. Implementation (B) — the stochastic element + typed addressable attention (LANDED)

Two pieces, both gated and dark by default.

**The stochastic element.** `ReadingAttention.superposition_scale(t) = 1 −
clamp(t, 0, 1)` (mirrors `Language.superposition_scale`) now scales the
**preference logits before masking** in both `ReadingAttention` and
`GlobalAttention`: `t=0` (pass A / no `<learning>`) → `×1.0` → sharp/exploit,
**byte-identical**; `t=exploreTemperature` (pass B) → flatter → exploration.
Scaling *before* the mask keeps the `−∞` coverage mask intact at any temperature
(the flat distribution is over the *legal* candidates only). The temperature is
the same one the grammar chooser already uses: `_set_superposition_temperature`
now also stashes `self._superposition_temperature`, which `_reading_attention_
step` / `_global_attention_step` read (None → 0 → byte-identical). No new flag —
it activates with the existing two-pass `<learning>`.

**Global attention (`Spaces.GlobalAttention`, `<globalAttention>`).** A free,
content/relation-driven attention that ranges over a **typed addressable space**
— a registry gathered by `BasicModel._addressable_spaces`:

| space | keys | shape | where |
|---|---|---|---|
| INPUT window | pooled per-span percept content | `[B, K, D]` per-batch | the span `[start,end]/N` |
| STM | `cs.stm.snapshot` rows | `[B, depth, D]` per-batch | per-row |
| LTM | `symbolicSpace.ltm_store.slots[:count]` pooled | `[M, D]` **shared** | per-row |
| CODEBOOK | symbol/whole codebook (`getW()`, WS-then-PS); meronomy+taxonomy | `[V, D]` **shared** | per-row; `boosts` = the codebook-retrieval prior |

One temperature-scaled softmax competes **across all spaces** (no monotonic mask
— free to land anywhere), emitting a **typed `.where`** (`space_id` +
`[start,end]`) and a **soft-read** `Σ αₖ·keyₖ`. Pointing the `.where` at the
codebook/LTM is *introspection / recall*; at the input window it is *reading /
search* — one mechanism, the type tag says which. **Shared stores stay `[M, D]`
and are matmul'd** (`α @ keys`), never broadcast to `[B, M, D]`, so a 65 536-row
codebook costs an `[B,V]·[V,D]` matmul, not a `[B,V,D]` tensor. **Gradient stops
at the keys** (codebook/LTM rows are EMA/persistent, detached); only the scorer
readout + the per-space `space_bias` train (the soft-read is differentiable
through `α` only). `_global_attention_step` is `@torch.compiler.disable`'d (host
gather) and **parks** the result on `_global_attention_obs` — the soft-read is
**not fed back** into the output this slice, so the forward is byte-identical
even with the flag on (the consumer that feeds the read into the task, and book
paging, are the deferred §7-item-4).

**Config:** `data/MM_global.xml` (MM_reading + `<globalAttention>true`).
**Tests:** `test/test_global_attention.py` (the module: cross-space ranging,
65k-row memory-safety, soft-read + typed `.where`, gradient boundary, the
temperature explorer; and the model wiring incl. the LTM-store path + optimizer
membership + temperature threading). The stochastic element's temperature on
reading is covered in `test_reading_attention.py`.

---

## 8. Open questions / decisions

- **Index space** — separate part/whole, each link carrying a part/whole type
  bit the route_hint reads. **(decided)**
- **Gradient boundary** — reading loss stops at the primed symbols; codebooks
  stay EMA/occurrence; codebook influence is occurrence-mediated. **(decided)**
- **Soft vs hard span selection** — start soft (`Σ αᵢ·spanᵢ`); harden to a
  straight-through argmax only if a diffuse early-training `α` smears PS's focus.
- **Stochastic primitive** — extend the two-pass soft-superposition (chooser in
  the gradient path) vs ergodic noise. Lean two-pass.
- **Relative-mask under parse-tree-deletion** — make it symbol-label-driven
  (STM symbol repertoire) rather than grammar-rule-driven, so decode needs no
  chart.
