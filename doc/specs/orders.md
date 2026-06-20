# The three order axes — `subsymbolicOrder`, `symbolicOrder`, `syntacticOrder`

*Spec authored 2026-06-19; resolves the SPECIFICATION GAP flagged in
[Architecture.md](../Architecture.md) "The orders".*

Three independent integer knobs govern how a forward pass composes
representations. They are **orthogonal axes**, not a single hierarchy: each
has its own semantics, its own bound, and a defined way of composing with the
other two. This document is the precise, separate spec the Architecture.md
callout asked for.

| knob | axis | unit | bound | default | parallel/serial |
|---|---|---|---|---|---|
| `subsymbolicOrder` (`T`) | analysis/synthesis **refinement passes** + area of attention | passes (CS→PS/WS pumps) | unbounded (practically small) | grammar-derived | applies to both modes |
| `symbolicOrder` | **parallel whole-slab** vs **serial σ/π abstraction** | mode selector | `{0, ≥1}` | grammar-derived (`1` if a non-substrate rule is enabled, else `0`) | `0` = parallel, `≥1` = serial |
| `syntacticOrder` | parse-tree **composition depth** | tree levels per sentence | **≤ number of words** in the sentence | `0` (inert ⇒ unbounded) | refines the serial path |

The one-line division of labour: **`subsymbolicOrder` says how many times to
re-look (and where); `symbolicOrder` says whether to abstract serially or
process the slab in parallel; `syntacticOrder` says how deep the parse tree may
grow when abstracting.**

---

## 1. `subsymbolicOrder` (`T`) — refinement passes and the area of attention

`T` is the number of **analysis/synthesis refinement passes**: codes are
passed back to PartSpace / WholeSpace across `T` iterations (the CS→PS/WS
loop). In PARALLEL mode (`symbolicOrder == 0`) the body walks `T` pre-built
stages (`body_stages`, `_forward_per_stage`); in SERIAL mode the same `T`
subsymbolic pumps run per word inside the word loop.

**Per-pass move (refine vs raise).** Each pass is one of two moves, which CS
chooses *per representation* by reading the **contiguity of `.where`** (the
endpoint-sum bracket `[start, end]`, `WhereEncoding.decode_span`):

- *contiguous* extent ⇒ **refine granularity** — chunk finer / tile (drive the
  radix), staying at the same abstraction order;
- *discontiguous* extent ⇒ **raise order** — another σ/π fold, lifting out of
  `.where`/`.when`.

The number of contiguous runs in a whole's `.where` *is* its part/whole ratio,
so the same read routes integrate-vs-disintegrate (→PartSpace σ vs
→WholeSpace π). See
[mereological-order-raising.md](mereological-order-raising.md) "The
three-aspect loop".

**Area of attention.** Attention scopes *what* each pass expands or chunks. The
mechanism is a **`.where` scope on the dual-input SECOND ARGUMENT** to PS/WS: a
null-concept event carrying only a `.where` (not content).

- **Default / model-driven** — a **full `.where`** lets PS/WS range over the
  whole input; attention then dives to a chosen location + granularity.
- **Override / deterministic reading** — the serial-word loop supplies **word
  `.where`s** as that second argument, forcing word-by-word reading.

One channel, two scope sources: `subsymbolicOrder` attention and the serial
(`syntacticOrder` / `symbolicOrder`) reading are the **same** mechanism
parameterized by *which* `.where` flows in. Reuses the existing dual-input
second argument + the `.where` brackets — no radix-filter rewrite. The
top-down WS→PS pass-back that implements this for the attention case is
specified in [mereological-order-raising.md](mereological-order-raising.md)
"the top-down attention handoff" (gated `<mereologyRaise>`).

`subsymbolicOrder` also sets the **max abstraction order** for the
ramsification table (`max(1, subsymbolicOrder)`).

## 2. `symbolicOrder` — parallel slab vs serial σ/π abstraction

`symbolicOrder` selects the **forward-dispatch shape**, NOT a pass count:

- `0` = **PARALLEL** — the whole-slab path. The per-stage body runs via
  `_forward_per_stage`; no per-word serial loop. The σ/π abstraction does NOT
  compose words one at a time; the slab is processed in parallel and the
  subsymbolic pumps refine it.
- `≥1` = **SERIAL / GRAMMATICAL** — the abstraction-composing path. The body
  runs `_forward_body_per_word` (one `IS_t` per word, push to STM per word) and
  the grammar composes the words with σ/π. Values `>1` are plumbed but behave
  as `1` today.

The knob is otherwise **untouched by this spec** — its read site, its
grammar-derived default (`1` when the XML enables a non-substrate rule, else
`0`), and its `_symbolic_order` stamp onto the WholeSpaces are all as before.
The precise distinction this spec pins down: `symbolicOrder` answers *"does the
σ/π abstraction run serially, or is the slab processed in parallel?"* — it is
the parallel-vs-serial switch, nothing more.

## 3. `syntacticOrder` *(new)* — parse-tree composition depth

`syntacticOrder` is the **maximum depth of the parse tree** built when the
grammatical (serial) path composes a sentence — a *distinct* axis from the
other two (composition *depth*, not refinement-pass count nor
parallel-vs-serial).

- **Bound.** A binary-branching parse over `W` words has at most `W − 1`
  internal composition levels, so the depth is **guaranteed ≤ the number of
  words** in the sentence. The cap is enforced as a **static** loop bound
  `min(syntacticOrder, cap − 1)` on the NULL-seal reduce sweep (`cap` = the STM
  capacity), so the trip count never depends on the runtime word count — the
  CUDA-graph / static-unroll contract is preserved. The **`≤ W` bound holds
  structurally**: a reduce micro-step is a no-op once a row's depth reaches 1,
  so a cap larger than the live depth still collapses fully, and a cap below it
  hands on a partially-composed forest of the remaining depth.
- **Leaves are words.** Synthesis halts at the **basic level** — word
  boundaries from analysis, consumed by the serial-word loop — so
  `syntacticOrder`'s leaves are words. It never tries to compose below a word.
- **Default `0` = inert ⇒ unbounded.** With `syntacticOrder == 0` (the
  default), the grammatical reduction runs to its natural fixpoint exactly as
  today (byte-identical). A positive value **caps** the number of σ/π
  composition levels the chart reduction may take (a depth budget): the
  reduction stops after `min(syntacticOrder, W)` levels and the partially
  composed forest is handed on. This is a strict refinement of the serial path
  — it never engages in parallel mode (`symbolicOrder == 0`), where there is no
  per-sentence parse tree to bound.

`syntacticOrder` **layers over** the serial `symbolicOrder` loop; it does not
subsume it. `symbolicOrder` decides *whether* to abstract serially;
`syntacticOrder` bounds *how deep* that abstraction may go per sentence. When
`symbolicOrder == 0`, `syntacticOrder` has nothing to bound and is inert.

## 4. How the three compose

For a serial/grammatical run the nesting is:

```
for each sentence:                         # discourse
  for each word w in the sentence:         # symbolicOrder >= 1  (serial read)
    for pump in 1..subsymbolicOrder:       # subsymbolicOrder    (refine/raise w)
      refine-or-raise(w)                   #   move chosen by .where contiguity
  compose the words into a parse tree      # symbolicOrder >= 1  (grammar)
    bounded to min(syntacticOrder, cap-1)  # syntacticOrder      (depth budget)
      fold levels (0 = unbounded fixpoint) #   leaves = words; <=W structural
```

- `subsymbolicOrder` runs **per node** (per word / per chunk) as the
  refinement-pass count.
- `symbolicOrder` is the **outer mode**: at `0` the word loop and the parse
  tree collapse to the parallel whole-slab path and only `subsymbolicOrder`
  pumps run; at `≥1` the serial word loop + grammar compose.
- `syntacticOrder` bounds the **depth of the grammar composition** within a
  serial run; in parallel mode it is inert.

The **basic-level stop** is shared: synthesis halts at words, so both the
serial word loop and the `syntacticOrder` tree bottom out at word boundaries
produced by analysis.

## 5. Configuration

```xml
<architecture>
  <subsymbolicOrder>3</subsymbolicOrder>   <!-- T refinement passes -->
  <symbolicOrder>0</symbolicOrder>         <!-- 0 = parallel, >=1 = serial -->
  <syntacticOrder>0</syntacticOrder>       <!-- 0 = unbounded; else <= #words -->
</architecture>
```

`syntacticOrder` is a non-negative integer, default `0`. It is read once at
build (`architecture.syntacticOrder`), stored on the model
(`self.syntacticOrder`), and consumed by the NULL-seal reduce sweep
(`_stm_reduce_to_single_S`) as a **static** per-sentence fold-level budget
`min(syntacticOrder, cap − 1)`; the `≤ W` bound holds structurally (a fold
no-ops once a row's depth reaches 1). Default `0` keeps every existing config
byte-identical.

---

## 6. The attention substrate (design — 2026-06-19)

> **Status: design / roadmap.** §1–§5 are implemented (`syntacticOrder`; the
> gated `<mereologyRaise>` top-down handoff in
> [mereological-order-raising.md](mereological-order-raising.md)). This section
> records *where the three orders are headed*: a single connectionist attention
> substrate over which the orders are **pump counts**, which replaces the
> explicit serial reading for-loop with learned reading.

### Attention is connectionist spreading activation, not QKV

The `<attention>` modes (`primer` / `second-order` = `A_Sᵀ diag(r_S) A_S` /
`low-rank`) are taxonomy-heat *retrieval*: activation spreads over the
codebook/lattice from primed rows. The reading/selection mechanism extends this
— it adds the **symbol + relation table** (below) as more graph to spread
through, and lets the spread **settle** over several pumps. The number of
settling pumps *is* an order knob. Priming is the existing `set_intent` →
`intent_boosts` → `selection_boost_fn` path (a `[V]` bias against the live
codebook rows), extended with a second prior from the relation store.

### The priming hierarchy is cumulative

Priming is layered, and each order sees every order **below** it (maps to
`abstraction_order` / the ramsification table; max order ≈ `subsymbolicOrder`):

- **first order — mereological entries.** Parts and wholes, grounded in
  `.where` (contiguous environmental extent). Primed bottom-up from what's seen.
- **second order — relations / concepts.** Built *on* the first-order
  mereological entries (a concept is a relation over parts/wholes).
- **higher order — relations of relations.** Sees *both* the mereological
  features *and* the lower-order relations/concepts, and composes on top.
  Concepts of concepts, with **no mereological `.where`** of their own.

Because higher-order features have no contiguous environmental extent, the
mereological (first-order) priming — which keys off `.where` contiguity (the
`RunStructureLayer` route) — **structurally cannot surface them**. They enter
attention only by spreading activation through the *relation* graph.

### subsymbolicOrder pumps mereology; symbolicOrder pumps relations

This is the precise content of "does the σ/π abstraction run" (§2):

- **`subsymbolicOrder`** = pumps over the **mereological** substrate (grounded,
  `.where`-having; primed by what's seen).
- **`symbolicOrder`** = pumps over the **relational** substrate (abstract, no
  environmental `.where`; reachable only through relation-of-relation edges).
  σ/π *is* this relational pump; without it the higher-order features get zero
  attention. `symbolicOrder = 0` attends only to environmentally-grounded
  features; `≥ 1` *also* pumps the relation graph to drag in the abstractions.

The §6c **sentence protocol** is already this shape: the parallel
`subsymbolicOrder` prelude commits intent-only (the gist primes both towers via
`set_intent`, nothing enters the workspace), *then* the serial per-word loop
runs — **prime subsymbolically from what you see, then pump symbolically**.

### Two kinds of `.where`

- **environmental / mereological `.where`** — a span over the input (the
  endpoint-sum bracket; what the #4 handoff scopes, what reading directs).
- **relational / symbolic `.where`** — a position in the symbol graph, for
  abstractions that don't live in the input. Rides on the
  non-contiguous-`.where` machinery (`part_chain`, `invert_ramsified`), not on
  surface spans.

### The symbol table: indices, owned by CS

The "wide" form (vs the "deep/tall" full codes that σ/π fold and the invertible
bind carries) is the **symbol table**, owned by ConceptualSpace. It stores
**indices, not vectors** — ownership-by-reference: the PS/WS codebooks own the
part/whole prototype vectors; the table owns only structure. A symbol's linkage
is captured from `PS.active` and `WS.active` at mint, kept as **separate
part-index and whole-index spaces** (each link carrying a part/whole type bit
the σ-refine-vs-π-raise routing reads); `materialize()` gathers the referenced
rows on demand (no duplication; auto-tracks codebook growth/retire). This is the
existing CS relation machinery unified — `create_word_object_meta`'s A/B/C
(word/object/reify-meta), the MetaSymbol category codebook, the
`RelativeTruthStore` — given an attention head.

### Learning: gradient stops at the symbols; codebooks move by occurrence

The reading loss (text mode: penalise attention that does **not** land on the
next word) trains the attention *readout*, with the **primed symbols as input**
— it does NOT backprop into the codebooks, so the EMA-only / detached-rows VQ
contract (C-9/C-11) stays intact. A symbol's importance reaches its codebook
rows *indirectly*: attending a symbol activates its referenced rows → those rows
occur → EMA consumes the occurrence. So "future importance" flows down the index
references as occurrence, not gradient — free, given the indices-not-vectors
design. The relations may take error directly (they are not VQ-constrained).

### The loop *is* the orders

Each pass: **prime** the attention from the current substrate (what's seen +
what's known) → **pump** spreading activation (subsymbolically over mereology,
symbolically over relations) → select the `.where` → scope PS/WS through the #4
handoff → emit → the substrate **updates** by occurrence (+ error on the
relations). `subsymbolicOrder` is the mereological pump count, `symbolicOrder`
the relational pump count, `syntacticOrder` the parse-tree depth the relational
pump may compose to. Reading — the `.where` attention trained by the next-word
loss — replaces the deterministic serial cursor and produces the
`_passback_scope_where` the #4 handoff already consumes.

The **build plan** for this substrate — reading attention (`<readingAttention>`),
global attention + the stochastic element it requires, and decoding an idea into
words via the grammar `.reverse()` with the parse tree deleted — is
[reading-attention.md](reading-attention.md).
