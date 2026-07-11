# Mereological Order-Raising

*Design spec. How the model builds a meronymic lattice over the two towers, assigns
objects at abstraction order 0, and raises order as necessary — driven by attention
(the gradient), converging to a stable lattice. Code-anchored; verified seams are
marked, gaps are marked ✗ (must build). Companion: [SymbolFirewall.md](../SymbolFirewall.md)
(this is the firewall's promotion invariant #7, made mereological).*

## Summary

The architecture has **two mereological towers** and a **symbol table** that bridges
them. The towers stay *in-kind*; the symbol is the *cross-tower link*:

- **σ — PartSpace.** *Parts compose parts*: bottom-up SYNTHESIS, the additive/union
  fold over atoms ([Spaces.py:8528](../../bin/Spaces.py), [8577](../../bin/Spaces.py);
  PartSpace owns only `self.sigma`).
- **π — WholeSpace.** *Wholes analyse wholes*: top-down ANALYSIS, the
  product/intersection fold over the unity ([Spaces.py:8530](../../bin/Spaces.py),
  [13620](../../bin/Spaces.py); WholeSpace owns `self.pi`).
- **Symbol — the META node.** A symbol is introduced to **link an overlapping part
  and whole**, and is associated with **both**: `insert_meta(ps_pos, ws_pos)` binds a
  PartSpace percept (a part, extension) to a WholeSpace symbol (a whole, intension)
  ([Spaces.py:14743](../../bin/Spaces.py); minted in perception by
  `_maybe_autobind_meta`, [Spaces.py:12285](../../bin/Spaces.py)).

The **corpus callosum** (in ConceptualSpace) is what *builds* the single meronomy
from the two towers: it links a part to a whole (`A isa B`, token `isa` type) when
their `.where` extents nest, bridges word↔object with a second-order meta-object, and
stores the result in symbol-relation LUTs that **are** the TruthLayer's absolute /
relative truth tables (see "The corpus callosum" below). Order-raising then maintains
that meronomy by three balancing forces, tracking **abstraction order** so abstract
(higher-order) concepts — which lose their `.where`/`.when` — keep their meronymic
relation to their lower-order parts. The **part/whole ratio** is the correctness
signal (and the MM_20M mean-collapse fix): a wrong ratio in some `.where` requests
further synthesis (σ) or analysis (π) there until the granularity is right.

## Principles

1. **Object identity is isomorphic.** An object's part-side (σ, synthesized up from
   atoms) and whole-side (π, analysed down from the universe) representations
   correspond. σ-up meets π-down **at the object**; the symbol records that the part
   and the whole are the same object.
2. **At the beginning there are only atoms and the universe.** PartSpace starts with
   **atoms** (the finest parts); WholeSpace starts with the **universe** (the single
   totality). The InputSpace emits these two views — the **Atom** (PartSpace
   synthesizes it) and the **Universe** (WholeSpace analyses it).
3. **Objects emerge from attention.** Repeated presentation / attention creates an
   object that has a *consistent* set of parts (synthesized) and a *consistent* whole
   (analysed). The object is what is stable on both sides.
4. **Synthesis and analysis are generally contiguous.** An order-0 object's parts
   form a contiguous extent (`.where`) over a contiguous interval (`.when`).
   **Discontiguity is what forces a higher order**: a concept whose instances are
   scattered in space/time cannot be a single contiguous extent, so it must be
   abstracted. **Higher-order parts/wholes are abstract — they lose `.where` AND
   `.when`.**
5. **Refinement is as-necessary and gradient-guided.** The meronomy is refined only
   when needed, under the pressure of attention (the loss gradient), and it
   **converges**: at some point no further refinement helps — a "good" number of
   parts and wholes for the lattice.

## The three balancing forces (a single meronomy)

The symbol table is kept a single coherent lattice by three forces; together they
converge it (principle 5).

1. **Link the tightest relation (order 0).** Associate the **largest** known part
   (PartSpace, via `.where` extent) with the **smallest** known whole (WholeSpace),
   connecting extension to intension. **Skip the link** when a *bigger part* or a
   *smaller whole* already subsumes the relation — never store a redundant edge.
   **Useless symbols are dropped.**
2. **Too many parts → synthesize a higher-order part.** When one whole accumulates
   many parts (`taxonomy_children`, [Spaces.py:14972](../../bin/Spaces.py)),
   synthesize a higher-order **part** (σ) that subsumes them, link it to the whole,
   and prune the now-moot individual part-whole edges. The result is a taxonomy
   linking each property (whole) to the largest-part / highest-order concept it is
   reliably associated with — psychologically the *basic level*.
3. **Only one part → Lewis' Singleton → analyse the whole.** A whole whose only part
   is itself is the trivial singleton; the whole should be **further analysed** (π
   divides it into finer wholes) — or, if the association was spurious, the link is
   **dropped**.

## The corpus callosum: one meronomy from two towers

Parts and wholes are *meant* to grow and shrink — bytes synthesize up toward words,
the whole analyses down toward words. **When they don't, that is a failure of the
chunker (σ) / analyser (π), not of the design.** Two consequences for the methods:

- **Synthesis (PartSpace σ) must climb.** BPE *should* merge bytes into larger
  parts; the **radix trie** (`_radix_token_stream` / `RadixLayer` / `PerceptStore`,
  [Spaces.py:7658](../../bin/Spaces.py)) is probably the best synthesis method —
  it grows parts by shared prefixes through use.
- **Analysis (WholeSpace π) should be restructured as PROPERTIES.** An analytic
  operation is one that produces a **binary tiling** of the input: the `.what`
  (and `.where`) that *have* the property vs. the `.what`/`.where` that do *not*.
  The `.where` is thereby partitioned. A "whole" analysed by a property splits into
  the region that satisfies it and the region that does not — that is how π descends
  from the universe toward words.

**The corpus callosum (in ConceptualSpace) is what joins the two towers** (the
learned `[2N, N]` `self.callosum` glue, [Layers.py:1703](../../bin/Layers.py);
[Spaces.py:8548](../../bin/Spaces.py), [12973](../../bin/Spaces.py)). At the callosum
a part `A` (from PartSpace) and a whole `B` (from WholeSpace) carry `.what` codes
from **different codebooks** — *not* directly comparable — but their **`.where` IS
comparable**. So the callosum asks `WhereEncoding`
([Spaces.py:353](../../bin/Spaces.py)): **is `A.where` a part of `B.where`?**
(the `.where`-containment measure is the ✗-build item). When that holds **and no
strictly greater part or lesser whole intervenes** (read off the codebook
activation), the callosum **links `A isa B`** — *token isa type*. The type becomes a
**name** for the token (a property set that picks out that token specifically). This
is how the callosum **builds a single meronomy out of the two towers.**

Worked example — the callosum links, at the same `.where`/`.when`:

```
A: the word "cat"   <= [ word()  & c & a & t ]
B: a cat object     <= [ animal() & orange() & object() ]
```

The link holds because the two **co-occur** at the same `.where`/`.when`. If one
occurs without the other, the link **weakens** — which is why the codebook coupling
wants a **Hebbian-like rule** (fire-together → wire-together; fire-apart → weaken).

### Word ↔ object: the second-order meta-object

The callosum links part↔whole *within* a modality, but it will **not** link the
*word* to the *object* — they are too unlike: **no convex set in conceptual space is
specific enough to associate the word with the object and rule everything else out.**
Linking them needs a **second-order concept**. So the cat-word and cat-object
representations are passed back to **PartSpace to synthesize a new whole** — a
**meta-object that lives OUTSIDE space and time** (no `.where`/`.when`; it is abstract
per Principle 4). That meta-object fuses the word-code and the object-code into the
single symbol used during **symbolic communication (serial mode)** — exactly the
MetaSymbol of [Language.md](../Language.md) (word ≡ object).

This is what *"object identity is isomorphic"* means precisely: the space a symbol
identifies is **bounded by its parts and wholes**, and those bounds *are* its
identity in conceptual space. The linking symbol — although it looks like a
`part()` operation over the parts/wholes it attaches — **establishes an identity for
both** the cat-word and the cat-object; tying the two symbols together **conflates
their identities** (one represents the other).

### Storage: symbol-relation LUTs == the TruthLayer

Because symbols/representations are **outliers** (abstract, no `.where`/`.when`,
codes from a symbol codebook), part↔whole associations are best kept **not** in the
meronymic taxonomy but in a **two-code-sized LUT**, and the symbolic taxonomy is
then **relations over symbolic indices**:

```
A: the word "cat"  <= [ word()  & c & a & t ]        # a proposition (object def)
B: a cat object    <= [ animal() & orange() & object() ]  # a proposition
C: B <= A                                            # a relation between symbols
```

`A`/`B` (propositions) and `C` (a relation over propositions) correspond **exactly**
to the TruthLayer's **absolute truths** (the consistent proposition corpus,
[Layers.py:6637](../../bin/Layers.py)) and **relative truths**
(`RelativeTruthStore` — "relations between two ideas and a relation",
[Layers.py:7071](../../bin/Layers.py)). **These two tables and the meronomy LUTs are
the same structure and must be integrated** (the part↔whole LUT *is* the absolute
table; the symbol-taxonomy relations *are* the relative table).

### Relations are learned explicitly (language → truth)

A symbolic-taxonomy relation is **learned from language and truth-evaluated**:
"cats are furry" is three words; symbolic lookup translates it to the relation
`[cats] <= [furry]` (a relative truth between two objects). So **words have an
explicit, truth-evaluable meaning**: if the proposition is trusted, it creates an
entry in the relative-truth table. The taxonomy is thus grown by *trusted
statements*, not only by perceptual co-occurrence.

### The part/whole ratio — and the MM_20M fix

The same machinery diagnoses and fixes the MM_20M mean-collapse. The meronomy is
**correct when the part/whole ratio is right**:

- **Many parts → one whole = UNDER-analysed** (the whole should be split / the parts
  re-synthesized into an intermediate whole).
- **One part → one whole = OVER-analysed** (Lewis' Singleton; collapse the level).

So an **incorrect part/whole ratio is the criterion** for requesting further
**synthesis or analysis within the problematic `.where`**. MM_20M collapses because
its byte chunker never climbs (8192 bytes → 8 fixed 1024-byte *blocks*, never words)
and its analyser never descends (`analysis=byte` keeps the whole as one chunk) — so
the ratio is pathological (one giant whole, no word parts) and nothing requests the
refinement that would build word-granularity parts for XOR. The fix is to let σ
climb (radix) and π descend (property tiling) until the ratio is right.

## The three-aspect loop: contiguity routes refine vs. raise

The pass-back from CS to the towers is not one operation but **three aspects**, and
CS chooses among them *per representation* by reading the **contiguity of `.where`**.
This splits the single `<subsymbolicOrder>` loop (which today does one undifferentiated
thing per pass) into two distinct subsymbolic moves, plus the serial symbolic loop:

1. **Refine granularity — subsymbolic, CONTIGUOUS.** A contiguous `.where` extent is
   a *localized region with internal structure*: CS passes it back so the tower
   builds **new parts/wholes by further analysis/synthesis** — σ chunks finer (drive
   the **radix trie**), π tiles by a property (the analyser dual). Stays at the same
   order, in space.
2. **Raise order — subsymbolic, DISCONTIGUOUS.** A discontiguous `.where` is the
   hallmark of abstraction (Principle 4): a scattered set has no single extent, so the
   only way to represent it as one thing is **another σ/π fold that lifts it out of
   `.where`/`.when`** — i.e. an order increase. Discontiguity → raise.
3. **Serial processing — symbolic.** The per-word serial loop (`symbolicOrder ≥ 1`).

The **zero `.where` vector is null** (no localization to act on — skip).

**Contiguity-structure *is* the part/whole ratio.** A whole's `.where` is some number
of contiguous *runs*, and the run-count *is* the part-count:

- **many runs under one whole** = too many parts = under-analysed → **disintegrate
  the whole** (π splits it at the gaps, on the WholeSpace side) *or* **integrate runs
  that belong together** (σ merges them, on the PartSpace side). CS routes the
  problematic `.where` to the tower that corrects the ratio.
- **one run** = one part = Lewis' Singleton = over-analysed → collapse a level.
- **fully scattered** (runs that are themselves structureless singletons) = the
  discontiguous case → **raise order** (aspect 2) rather than re-split.

So a single `.where` read drives everything: refine-vs-raise (aspects 1 vs 2),
integrate-vs-disintegrate (which tower), and *which* parts to feed back ("the parts
of interest" = the `.where` regions whose run-count is wrong — this is how the corpus
callosum **drives the radix synthesis**).

**The one new primitive** this needs: contiguity is *not* a property of a single
object's `.where` but of the **set of its constituents'** positions — so the callosum
needs, per candidate, the **extent + run-structure** over constituent `.where`s (how
many gaps). That is a strict superset of the `.where`-containment measure; everything
else above is routing on top of it.

> **Substrate landed (2026-06-16).** `.where` (and `.when`) are no longer single
> quadrature points: they are **endpoint-sum brackets** `[start, end]`
> (`WhereEncoding.encode(start, end)` / `decode_span`, angle = center, magnitude =
> extent; an instant snaps to zero extent). So a *single* code already carries an
> extent, and the **`.where`-containment query** below is a direct read
> (`A.start ≥ B.start ∧ A.end ≤ B.end`). The remaining run-structure work is only the
> aggregation over a *set* of constituent brackets (counting gaps between sibling
> spans); the per-code extent/containment is built. See [doc/Spaces.md](../Spaces.md).

### Top-down attention handoff: WS scopes PS via the passed-back chunk + `.where` (2026-06-19, Alec)

When attention is on the **"words"** category (top-down, from WS), the actions PS
takes on *subsequent* `subsymbolicOrder` passes are scoped by what WS passes back
through the recursive (WS → PS) connection. **The first pass is wide-open** — PS
ranges over the whole input (model-driven attention); later passes narrow. What is
passed back selects the operation:

| passed back to PS | meaning |
|---|---|
| null chunk, **no `.where`** | **no-op** — nothing to attend |
| **null chunk + the `.where` of the nth word** | **focus on the parts of the nth word** — the scoped parse (this *is* the `.where`-scope-on-the-second-argument mechanism; see [Architecture.md](../Architecture.md)) |
| **multiple chunks** | **chunk** — PS must analyse finer (σ refine, aspect 1) |
| **single non-null chunk** | **return that chunk + the parts under it** — no further chunking |

The serial-word override (`syntacticOrder` / `symbolicOrder ≥ 1`) drives the same
connection deterministically — supplying the word `.where`s serially instead of
letting attention choose (the same one mechanism, per the Architecture.md callout).

**Word identity, and the higher-order trigger.** The word to identify is **unique**:
it has a particular **union of parts** *and* a particular **union of wholes** —
identity is those two unions converging on a single symbol. If a candidate symbol has
**disjoint sets of wholes and parts** (its parts do not sit under its wholes — the
discontiguous / abstraction case, aspect 2 above), it must be **higher-order**: run it
through **σ (SigmaLayer) or π (PiLayer)** to produce a better approximation *under the
whole*. So refine-vs-raise is keyed on **parts/wholes set-overlap** as well as `.where`
contiguity — disjoint ⇒ raise.

**Remaining work (handoff):**
- **Pass-back dispatch** — interpret the WS→PS recursive payload by the table above
  (no `.where` → no-op; null + word-`.where` → scoped parse; multiple chunks → σ
  refine; single non-null chunk → return chunk + sub-parts).
- **First-pass-wide gate** — pass 0 ignores any scope (wide-open); scoping applies
  from pass 1 onward.
- **Null-chunk + `.where` second argument** — the (null content, `.where`) scope on
  the PS/WS forward (the mechanism flagged in Architecture.md), reusing the existing
  dual-input second argument + the `.where` brackets (no radix-filter rewrite).
- **Identity test** — `union(parts) ∩ union(wholes)` converging on one symbol;
  **disjoint ⇒ route through σ/π** for the higher-order approximation.
- **Substrate (blocker)** — the multi-stage carrier must survive `t>0` first: the
  wide↔deep-emit × combine-deep-expect reconciliation (the MM_20M `sO=3` bug). Today
  the `t>0` combine no-ops on the None return, so the recursive WS→PS connection has
  nothing to thread — this handoff sits on top of that fix.

## Higher-order parts

A higher-order part is created on the **PartSpace (σ) side**, because the part side
is where extension is synthesized and a code can **geometrically contain its
constituents**.

- **Synthesis over a set (✗ build).** Combine the M constituent part-codes into one
  higher-order code via PartSpace `self.sigma` ([Spaces.py:~8932](../../bin/Spaces.py)).
  Today `self.sigma` is a per-position fold; a *synthesize-over-a-set* op is new.
- **Subsymbolic first, explicit later.** Initially the higher-order code
  **discriminates** its constituents subsymbolically (its geometry contains them);
  once enough exemplars have trained it, the code stands on its own. We do the former
  until we have enough exemplars to train the higher-order code appropriately.
- **Relink + prune.** The higher-order part code (which now geometrically contains
  its parts) is linked against the WholeSpace code previously linked to many parts;
  the numerous part-whole edges that were necessary before are now moot and removed.

## Order tracking and the discontiguity problem

A higher-order part is abstract — its `.where`/`.when` are no longer contiguous (it
transcends a specific place/time). So its meronymic relation to its lower-order parts
**cannot be read off `.where`** and must be tracked explicitly:

- **Abstraction order** = the count of folds a code underwent: 0 = atom / proper
  noun / prototype, 1 = noun / type, 2 = count noun, higher = more abstract. Read by
  `abstraction_order(row)` ([Spaces.py:2272](../../bin/Spaces.py)) over the
  ramsification table ([Spaces.py:1860](../../bin/Spaces.py); FOLD_NEITHER / FOLD_SIGMA
  / FOLD_PI). The table applies to **both** towers.
- **Recording (✗ wire).** `record_fold` ([Spaces.py:2252](../../bin/Spaces.py)) exists
  but is **test-only** — it must be wired into the live σ/π folds so order is recorded
  during perception, and the table enabled at build (`enable_ramsification(max_order)`,
  `max_order ≈ subsymbolicOrder`).
- **Provenance sidecar (✗ build).** A `part_chain` record on the higher-order META
  row naming its constituents, so the meronymy survives a non-contiguous `.where`.
- **Inversion.** `invert_ramsified` ([Spaces.py:2282](../../bin/Spaces.py)) already
  walks the fold sequence in reverse — the basis for reconstructing a high-order code
  back to its constituents.

### The symbol table and the priming hierarchy (design, 2026-06-19)

The abstraction-order ladder above is also the **priming hierarchy** for attention —
see [orders.md §6](orders.md) "The attention substrate" for the full account. In brief:

- **first-order = mereological entries** (parts/wholes, `.where`-grounded, primed
  bottom-up from what's seen); **second-order = relations/concepts** built on them;
  **higher-order = relations of relations**, which see *both* the mereological features
  *and* the lower-order relations and have **no mereological `.where`** of their own.
  Higher-order features therefore cannot be primed off `.where` contiguity (the
  `RunStructureLayer` route) — they enter attention only via spreading activation
  through the relation graph. That relational pump is what `symbolicOrder ≥ 1` runs.
- So there are **two kinds of `.where`**: the *environmental* one (a surface span, what
  the top-down handoff scopes) and a *relational* one (a position in the symbol graph)
  for the abstract codes, which rides on the `part_chain` / `invert_ramsified`
  machinery above rather than on surface spans.
- The **symbol table** that holds these relation/abstraction edges is owned by
  **ConceptualSpace** and stores **indices, not vectors** (ownership-by-reference: the
  PS/WS codebooks own the prototype vectors; the table owns only structure). A symbol's
  linkage is captured from `PS.active` / `WS.active` at mint, kept as **separate
  part-index and whole-index spaces** (each link carries a part/whole type bit the
  σ-refine-vs-π-raise routing reads). It unifies the existing CS relation machinery
  (`create_word_object_meta`'s A/B/C, the MetaSymbol category codebook, the
  `RelativeTruthStore`) under one attention head. The reading/attention loss trains the
  readout but **does not** backprop into the codebooks (preserving the EMA-only VQ
  contract); a symbol's importance reaches its codebook rows indirectly, via use →
  occurrence → EMA.

## Words: the SymbolTable's double duty

A **word is a whole with TWO parts** — an **orthographic** part (the surface form)
and an **object** part (the referent) — because the word designates both itself and
its object. So each word is **two associations** in the symbol table:

- The `ReferenceTable` ([References.py](../../bin/References.py): `bind(word, obj)`,
  `_by_word`) and the SS-side `reference_table` ([Spaces.py:16575](../../bin/Spaces.py))
  hold word↔orthography and word↔object; `resolve_ps_terminal` resolves a perceptual
  terminal to its symbol.
- The **object code is random-init** for a text-only learner (`insert_symbol` seeds a
  fresh row uniformly) and is shaped by syntactic/semantic pressure. Corpus gradient
  lands on the *symbol* side, so a fresh object is under-trained — which is exactly
  why **order-raising on the object part** (subsuming it under a higher-order part
  grounded in already-trained codes) lets it inherit training.
- **Serial-mode meta-symbol = {orthographic code, object code}.** A word isolated in
  serial mode is automatically bound into a **higher-order meta-symbol** carrying the
  word code plus the object code (the MetaSymbol of
  [Language.md](../Language.md) "Participation Categories"; the META node uniting
  word≡object). Every word-mapping mints the object-mapping.
- **Substitution around the grammar (✗ verify/wire).** In serial mode, **identified
  words are replaced by their objects BEFORE syntactic composition, and objects are
  replaced by their words AFTER decomposition** — so `compose`/`reverse` operate on
  **objects**, not surface words. The pieces exist (`ReferenceTable`,
  `resolve_ps_terminal`); confirm the substitution is wired around the serial
  compose/generate path (`_forward_body_per_word`), and wire it if absent.

## Code map — EXISTS (reuse) vs ✗ (build)

| Piece | Status | Anchor |
|---|---|---|
| Parts compose parts (σ) / wholes analyse wholes (π) | **exists** | [Spaces.py:8528/8530](../../bin/Spaces.py), `self.sigma` / `self.pi` |
| Corpus callosum glue (joins the towers in CS) | **exists** | `self.callosum` `[2N,N]` [Layers.py:1703](../../bin/Layers.py); [Spaces.py:12973](../../bin/Spaces.py) |
| Symbol links part+whole (META) | **exists** | `insert_meta` [Spaces.py:14743](../../bin/Spaces.py); `_maybe_autobind_meta` [12285](../../bin/Spaces.py) |
| Enumerate a whole's parts | **exists** | `taxonomy_children` / `ps_children_of_whole` [Spaces.py:14972](../../bin/Spaces.py) |
| Abstraction order (read) | **exists** | `abstraction_order` [Spaces.py:2272](../../bin/Spaces.py) |
| Fold inversion | **exists** | `invert_ramsified` [Spaces.py:2282](../../bin/Spaces.py) |
| Word↔object table | **exists** | [References.py](../../bin/References.py); `reference_table` [Spaces.py:16575](../../bin/Spaces.py) |
| Radix synthesis (σ climbs by shared prefixes) | **exists** | `_radix_token_stream` / `RadixLayer` / `PerceptStore` [Spaces.py:7658](../../bin/Spaces.py) |
| Absolute / relative truth tables (== the LUTs) | **exists** | absolute set [Layers.py:6637](../../bin/Layers.py); `RelativeTruthStore` [7071](../../bin/Layers.py) |
| σ-synthesis over a SET (higher-order part) | **implemented (gated)** | `SigmaLayer.synthesize_over_set` [Layers.py](../../bin/Layers.py) (raise uses mean-combine first) |
| Provenance sidecar (`part_chain`) + persistence | **implemented (gated)** | `part_chain` + `vocab_extras` [Spaces.py](../../bin/Spaces.py) |
| "Many parts → one whole" detection + raise | **implemented (gated)** | `maybe_raise_order` in `_maybe_autobind_meta` |
| Relink + prune (link-removal API) | **implemented (gated)** | `delete_meta` / `unlink_child` [Spaces.py](../../bin/Spaces.py) |
| `record_fold` order stamp | **partial** | stamped at mint only ([Spaces.py:2252](../../bin/Spaces.py)); live σ/π per-row attribution deferred |
| Singleton (one part) → analyse / drop | **partial** | count==1 path; no-op first pass |
| `.where` part-of containment query (callosum) | **substrate built (2026-06-16)** | `WhereEncoding.decode_span` [Spaces.py:353](../../bin/Spaces.py) → `(start,end)` interval; containment is `A.start≥B.start ∧ A.end≤B.end` |
| `A isa B` callosum linking (token isa type) | ✗ build | gate on `.where` containment + no intervening part/whole (codebook activation) |
| Hebbian codebook coupling (co-occur → wire) | ✗ build | strengthen/weaken the cross-codebook link on co-/dis-occurrence |
| Property-tiling analysis (π binary tiles `.where`) | ✗ build | analysis op = has-property vs not, partitions `.where` |
| Second-order word↔object meta-object | ✗ build | synth in PartSpace, outside `.where`/`.when` (the MetaSymbol) |
| Symbol-relation LUT ↔ TruthLayer integration | ✗ integrate | part↔whole LUT == absolute; symbol-taxonomy == relative ([Logic.md](../Logic.md)) |
| Part/whole-ratio criterion → request σ/π refinement | ✗ build | wrong ratio in a `.where` triggers further synthesis/analysis (MM_20M fix) |
| `.where` run-structure (extent + gap count over constituents) | ✗ build (per-code extent built) | the one new primitive; per-code extent via `decode_span` (2026-06-16); remaining = gap-count aggregation over a set |
| Factorize a π fold (log-domain dual of σ's balanced split) | ✗ build | `PiLayer` ([Layers.py:~3790](../../bin/Layers.py)) reverse via `W⁻¹` then split the recovered log-sum equally by `M` |
| Overcomplete tiling (match `.what`, many parts, paint/average) vs. partition | ✗ build | part tower returns many matches; IS paint/average recombination; over-collection lifecycle prunes |
| Basic-level bound (`.what`-size ∈ [min,max]; text = letter count) | config present, wiring ✗ | `architecture.basicLevel{Min,Max}Size` ([model.xsd](../../data/model.xsd); `MM_mereology.xml` = `0`/`24`) |
| Pass-back routing: contiguous→refine / discontiguous→raise / zero→null | ✗ build | CS reads run-structure, splits `<subsymbolicOrder>` into aspects 1 vs 2 |
| Integrate (→PartSpace σ) vs disintegrate (→WholeSpace π) by run-count | ✗ build | CS routes the wrong-ratio `.where` to the correcting tower |
| Serial word↔object substitution around compose | ✗ verify/wire | `_forward_body_per_word`, `resolve_ps_terminal` |

## How this reshapes Stages A and B

This spec comes **before** the XOR work because it changes it:

- **Stage A (parallel, MM_20M).** Names the two InputSpace subspaces correctly: the
  **Atom** (PartSpace synthesizes) and the **Universe** (WholeSpace analyses) — not a
  byte/word split. The `<analysis>word</analysis>` knob cuts the Universe into parts.
- **Stage B (serial, MM_20M_grammar).** Reconstruction runs on **object** codes (the
  word↔object substitution above): the reverses recover operand *objects*, and
  word-exactness is measured after the object→word swap. The order machinery also
  explains why some reverses are genuinely non-invertible (a predicate/reduction
  destroys the parts) — those are signed off, not faked.

## Integration + gate

Detection and raising run in the autobind / perception-end path, where
`record_lbg_pull` / `maybe_split_lbg` already run under `no_grad`
([Spaces.py:15073](../../bin/Spaces.py)). Gate the whole pipeline on a new flag (e.g.
`<mereologyRaise>`); `subsymbolicOrder` sets the max order. **Off by default ⇒
byte-identical** (verified by the full suite).

## Acceptance criteria

1. A fresh object mints at order 0 (contiguous `.where`/`.when`).
2. The order-0 link is the tightest: largest part ↔ smallest whole, no redundant edge.
3. Many parts of one whole → a higher-order part forms, geometrically contains its
   constituents, links to the whole, and the moot edges are pruned; its order = 1 and
   its constituents are recoverable via the provenance sidecar.
4. A singleton whole triggers further analysis (or the link is dropped).
5. Abstract codes carry no `.where`/`.when` but retain their meronymy via order +
   provenance.
6. Serial compose/reverse operate on objects; reconstruction regenerates surface
   words via the object→word swap.
7. The lattice converges (refinement stops at a "good" part/whole count).
8. Flag-off is byte-identical.

## Open design choices

- **Raise vs divide** when a whole has many parts: produce a higher-order part
  (recommended) vs divide the whole to reduce granularity.
- **Constituent selection**: subsymbolic geometric containment (recommended first) →
  explicit set once exemplars suffice.
- **"Many" / "good" thresholds**: config knobs; ultimately gradient-driven.
- **Well-representability of the trigger**: parts/whole counts are the primary signal
  (per this spec); `invert_ramsified` reconstruction residual is an auxiliary check.

## Reframing — two parthoods, two memories, two gates (2026-06-16)

A design discussion (with a cited cognitive-science check) sharpened where the `.where`
machinery applies and what the rest must be. Recorded here for later doc inclusion.

### Two parthoods

- **Mereological part-of — ORDER 0, `.where`-grounded, within one presentation.** paw ⊑
  this-cat; byte ⊑ "cat"; "New York" ⊑ the sentence. The part's extent nests in the whole's
  extent *now*. This is what `.where`-containment (`RunStructureLayer.contained_mask`) and
  the autobind binding (below) build. **The meronomy proper is over 0th-order parts/wholes.**
- **Taxonomic subsumption — SYMBOLIC, across presentations.** the class *cats* ⊑ the class
  *animals*. This is **NOT** `.where`-containment: classes range over **discontiguous
  prototypes** and have no single extent (Principle 4). It is a relation between abstract
  symbols, learned explicitly (trusted language) and/or by reasoning over examples, with the
  π intensional **meet-absorption** (`π(cat,animal)≈cat`) + `abstraction_order` as the
  geometric/order correlate — a *similarity weight*, not the source.

> **Cognitive grounding (deep-research, adversarially verified).** Superordinate categories
> cohere poorly perceptually (Rosch et al. 1976: ~0–3 shared attributes vs ~7–9 basic-level);
> a label glues perceptually-dissimilar things (Markman & Hutchinson 1984; Waxman & Gelman
> 1986) but does not *create* categories (Waxman et al. 1991); category-over-appearance
> induction is gradual (Godwin & Fisher 2015) and the theory-vs-similarity question is a
> hybrid (Fisher/Godwin/Matlen 2015); explicit class-inclusion is a late achievement
> (~8–9; Politzer 2016). Verdict: subsumption is **fundamentally symbolic / language-mediated
> / inferential**, not perceptual-similarity or spatial-containment — a qualified yes, hybrid,
> emerging developmentally. basic-level (the *word* level) is the perceptual/early rung;
> superordinate is the symbolic rung; **order-raising is the basic→superordinate trajectory,
> and the word/label is the bridge** (the word↔object MetaSymbol).

### The partition (codes vs relations × absolute vs relative)

- **Absolute / perceptual.** **Codes** live in the **PartSpace (σ) + WholeSpace (π)
  codebooks**; **relations** live in the **CS meronymy** (the part↔whole META taxonomy,
  `insert_meta`/`taxonomy`).
- **Relative / taxonomic.** **Relations** live in the **CS symbolic taxonomy** (symbol↔symbol,
  `RelativeTruthStore`).

### User truth, and the two gates

User-provided **Truth = English sentences**. At the surface they give **meronymic relations**
(the parse); composed syntactically they yield an **idea = absolute truth** (a proposition) or
a **relation between ideas = relative truth**. Where a truth lands is decided by **two gates**:

1. **Reducibility.** Can it be **encoded within the codebook** (reduced to codes + relations)?
   **Grammar is the reduction mechanism.** A sentence that **cannot be reduced to the codebook
   absent grammatical operations** must be **stored as an IDEA** (a composed proposition — not
   a code, not a meronymy edge). An *absolute* truth can itself be an idea, not only a relative
   one.
2. **Trust.** Is it trusted? Trust is the **tetralemma** `(t,f,b,n)` (TRUE/FALSE/BOTH/NEITHER;
   `meta_trust` / `insert_meta(trust=...)` / `RelativeTruthStore`).

**Asymmetry (key):** the **codebooks are assumed VALID — they carry NO per-proposition
DegreeOfTrust**. A proposition enters the codebook (and thereby the meronymy/taxonomy *as the
truth table*) only when it is **both reducible AND trusted**. **Untrusted and/or irreducible**
truths are **represented in the TruthLayer / as ideas**, carrying DegreeOfTrust. Hence
"CS meronymy == absolute-truth table" and "CS symbolic taxonomy == relative-truth table" hold
**only for the reducible-and-trusted portion**; the remainder lives as trust-annotated ideas.

### Three memories (all but the codebook trust-bearing)

- **Codebooks** (PS σ, WS π) — *codes*; assumed valid; no per-proposition trust.
- **CS meronymy + symbolic taxonomy** — *relations* (the absolute/relative truth tables for the
  reducible-and-trusted portion); trust-bearing.
- **Ideas / relations-between-ideas** — composed propositions that did not reduce; held in
  **STM** (transient, per-sentence) and persisted in **LTM** (episodic exemplar store);
  trust-bearing. LTM verifies a general (symbolic) relation by **retrieving order-0 instances**
  and checking parthood where `.where`/co-occurrence still hold (feeds the TruthLayer trust).

> **"LTM" disambiguation.** The code already uses *LTM* for the inter-sentence AR chain of STM
> end-states (`Layers.py` ~7365, discourse continuation). The store meant here is a NEW
> **episodic exemplar store** of persisted CS `.events` (or PS `.what`+`.where`+`.when`),
> distinct from PerceptStore (types), the codebooks (prototypes), and STM (transient).

> **✗ FUTURE TODO — provision LTM by pre-parsing a corpus.** WikiOracle is **stateless**, so it
> cannot accumulate memory across runs: LTM must be **provisioned (loaded), not learned
> online**. Pre-process a large database (e.g. **Wikipedia**) into an explicit **pre-parsed**
> format — sentences pre-composed into ideas (absolute) / relations (relative) with per-memory
> trust, reduced into codes + taxonomy where possible, else held as ideas — and load it as the
> model's LTM. (Fits the project name: an oracle that has ingested Wikipedia.)

### Analysis = property-tiling; the wholes are intensional TYPES (2026-06-16, Alec)

How WholeSpace chunks (π) — and why there is no per-sentence churn:

- **The first whole is "a sentence" — a generic TYPE, minted once**, not per-occurrence. Its
  spatial extent *is* the **Universe** (the entire `[0, N_raw)` input passed from InputSpace).
  So binding under it does not grow the codebook (no churn): one reusable Universe/sentence
  whole.
- **A property IS a binary tiling.** Applying a property — `word` (runs of letters vs
  non-letters), `whitespace`, `punctuation`, `digits` — splits the input into two wholes:
  `{has-property}` and `{¬has-property}`, each an **intensional** class carrying a **`.what`**
  (which class) AND a **`.where`** (the regions satisfying it). This is the analysis dual of a
  synthesis chunk.
- **`.index` selection is OR over properties (analysis), dual to AND over particles
  (synthesis).** Selecting several property rows via `SubSpace._index` unions their tilings:
  `letters OR digits` → `{letters ∪ digits}` vs `{¬(letters ∪ digits)}`. In PartSpace the same
  `_index` mechanism is an AND over particles (a part is the conjunction of its constituents —
  more particles narrow it); in WholeSpace it is an OR over properties (more properties widen
  the whole). Union-of-properties widens; conjunction-of-particles narrows.
- **Both towers chunk and return representations.** PS chunks by *synthesis* (merge particles →
  parts; radix spell-out); WS chunks by *analysis* (tile by property → wholes); both emit
  `.what` + `.where` codes.
- **Properties are TOKENS, and they range from localizable to GLOBAL/BACKGROUND.** A property is
  a codebook row (a token), selected via `.index`. Some properties are *localizable* (char-class:
  letters at positions 3–7) and tile into regions *with* a `.where`. Others are *global /
  background* characterizations of the WHOLE that belong to **no part** — the canonical example is
  the **frequency content of an image** (a property of the entire image, not any pixel). A global
  property has **no localized `.where`** (the Universe is its extent; abstract by Principle 4).
  This is exactly why `materialize_property`'s basis is **low-frequency / whole-ranging**: a
  property row dotted with the positional (frequency) basis IS a frequency/background
  characterization of the whole. So the property-token codebook + `materialize_property` + `.index`
  is **one unified API** spanning char-class tilings and global characterizations — char-class is
  just a content-keyed backend, not a separate mechanism.

### Both towers are over TYPES; the distinguishing axis is mereological support

The PS↔WS duality completes: **both towers deal in TYPES, not tokens.** A PS percept (the letter
`a`) is a *type* of which an image may hold **many tokens** (the PerceptStore already works this
way — a percept-id IS the type, its occurrences are uses); a WS property is likewise a type. The
**one thing that distinguishes a part-type from a whole-type is its mereological SUPPORT** — the
`.where`(s) of its instances:

- **PS type → extensional support**: the SET of its token-occurrences (AND-of-particles narrows
  the type; occurrences may be many and scattered).
- **WS type → intensional support**: the REGION it tiles (OR-of-properties widens the type).

Consequence: **a type whose support is many/scattered tokens has no single `.where` → it is
abstract / higher-order** (Principle 4), carried by its **footprint** (the support set). So "many
tokens of `a` → the type `a`" is the SAME order-raising move as "bytes → word": many parts under
one whole, raise a higher-order type, track it by its support footprint; the order-0 case is a
single contiguous token. Therefore the `.where` / run-structure / containment machinery
(`RunStructureLayer`: `contained_mask`, `n_runs`, `tightest_container`) **IS the mereological-
support computation** — the shared substrate both towers' types stand on, and the defining
feature that separates a part-type (extensional support) from a whole-type (intensional support).

### How analysis/whole types integrate with the `(ps,ws)` meronomy (2026-06-16, Alec — RESOLVED)

The integration question — "how do WholeSpace analysis/property wholes (which have `.where`
regions but are not PS percepts) connect to the `(ps,ws)` meronomy?" — is answered by the
type/instance/`.where` distinction:

- A WholeSpace **word** whole is ONE `.what` code (the word TYPE); the several word-instances in a
  sentence are that one type at different `.where`s — exactly as the PartSpace **`A`** is one
  `.what` code (the letter TYPE) with many `.where` instances. Neither is "actual" (a glyph in a
  font). The word-type may even **reconstruct better** than a letter-part (its constituent letters
  average to the type mean M) — the code is the stable thing.
- **The meronomy edge is TYPE → TYPE:** "is `A` a part of *word*?" means "do we store the PS
  `A`-code as a part of the WS word-code?" — a single edge between the two TYPE codes in the
  `(ps,ws)` taxonomy.
- **`.where` is the EVIDENCE that decides the edge, not the storage:** *"sometimes we do and
  sometimes we don't, and we know by looking at the `.where` for each."* The edge holds when the
  part's `.where` nests inside the whole's `.where` (cross-tower **`contained_mask`** /
  `tightest_container`: PS-part `.where` ⊆ WS-whole `.where`). So the per-instance `.where`s are
  the deciding evidence; the **codes (types) + their type→type edges are what persist** — instances
  are never stored as parts, so there is **no taxonomy churn** (this resolves the open
  instance/type ↔ STM/codebook integration point).

**Implementation consequence.** The binding is the cross-tower `.where`-containment ("atom-index
join"): compare PS-part `.where`s (`where_idx` byte offsets) against WS-whole `.where`s
(`property_spans` / `stage_analysis_spans` runs) via `contained_mask`/`tightest_container`, and
store the part-code ⊑ whole-code edge where it nests. The substrate is built (`property_spans`,
`where_idx`, `contained_mask`, `tightest_container`). NOTE: **A4-core bound per-instance run-span
positions — the wrong granularity**; the correct A4 binds the PS part-TYPE to the WS whole-TYPE,
`.where`-gated. `property_class_whole` (the generic WS class/"word" type) and the raise plumbing
are reused unchanged; only the binding granularity (per-instance → type→type, `.where`-gated)
changes.

### Ramsification with the towers — synthesize/factorize vs. categorize (2026-06-17, Alec)

Ramsification (fold a set into one code) runs on **both** towers, and the **contiguity** of the
support decides whether the fold is **invertible**:

- **PartSpace (σ).** **Contiguous parts → synthesize a higher-order WHOLE part.** The fold is
  **invertible**: `SigmaLayer.generate` does the balanced split (divide the summed membership
  equally across the `M` constituents), so the constituents are recoverable — the high-order code
  *geometrically contains* them. **Discontiguous parts → categorize a COMMON output vector.** The
  scattered constituents share one category code; this is **lossy / non-invertible** (the category
  alone cannot say *which* scattered constituents produced it). Contiguity is read from `.where`
  run-structure (`RunStructureLayer.n_runs`): one run ⇒ synthesize; many runs ⇒ categorize.
- **WholeSpace (π) — the dual.** **Analyse-into-properties** (the π fold) and its inverse
  **factorize**. A contiguous intensional region factorizes back into its sub-properties; a
  discontiguous one collapses to a common intensional characterization (lossy, the analysis-side
  mirror of "categorize").

**How to factorize a PiLayer's output.** `PiLayer` is the multiplicative / log-domain fold
([Layers.py:~3790](../../bin/Layers.py)): `z = _from_mult(exp(W·log(_to_mult(x)) + b))` with
`log(_to_mult(x)) = 2·atanh(x)`. The exact reverse exists when `W` is invertible. **`factorize_
over_set` is the log-domain DUAL of σ's balanced split:** recover the summed log-membership with
`W⁻¹` (and subtract `b`), then **split it equally across the `M` factors** (divide the recovered
log-sum by `M`) and map back through `_from_mult`/`exp`. Where σ splits a *sum* equally in the
linear domain, π splits a *product* equally in the log domain — same balanced-split idea, dual
algebra. In `monotonic` mode (`W ≥ 0`) the ordering is preserved, so the factorization is stable.

**Two structural decisions for high-order codes.**
1. **Contiguous blocks per order.** A high-order code reserves a **contiguous block** in `.what`
   so its constituents remain addressable as a run and the run-structure / containment machinery
   (`contained_mask`, `tightest_container`) keeps working on it. The codebook is partitioned into
   per-order contiguous regions rather than interleaving orders.
2. **A symbol table per order.** Each abstraction order keeps **its own** symbol table — order-`k`
   `Parts`/`Wholes` relations are within-order (or one order up), so order-`k` structure never
   collides with the order-0 identity ties. "The symbol table drives both towers + order-raising"
   is thereby naturally **stratified by order**.

### Overcomplete tiling bounded by the basic level (2026-06-17, Alec)

Instead of returning a **partition** of the input (every position in exactly one part), the part
tower returns an **OVERCOMPLETE TILING**: the `.what` codebook already lives in `subspace.what`, so
**match the input against `.what` and return MANY matching parts** — overlapping, redundant. The
overlaps are resolved by the existing **IS paint/average** recombination (see
[[processing-contract-spaces]]: paint each match into its `.where`, average the overlaps), and the
redundant matches are **pruned by the over-collection lifecycle** (the CS symbol table's
over-collection trigger retires the ones that don't earn their keep).

**Bound the overcomplete set by Eleanor Rosch's BASIC LEVEL — not top-`k`** (top-`k` is an odd
spec: it fixes a count rather than a meaningful grain). The bound is a **min/max SIZE in `.what`**,
where *size* = the count of **COMMITTED (`±1`) properties** = **specificity**:

- **all-`0`** = the **Universe** / maximally general (no committed property);
- **all-committed** = an **atom** / proper noun / maximally specific;
- the **mid band** = the **basic level** (Rosch et al. 1976: the level acquired first, maximally
  informative-yet-distinctive).

Keep matches whose `.what`-size ∈ `[min, max]`; drop the too-general (below `min`) and too-specific
(above `max`).

**Text-model bounds as model-spec properties.** For a TEXT model "size in `.what`" is simply the
**count of letters**, so a part ranges from **0** (the Universe) up to the longest reasonable word:
`len("disestablishmentarianism") == 24`. These are set as model-spec properties
`architecture.basicLevelMinSize` / `basicLevelMaxSize`
([model.xsd](../../data/model.xsd); `data/MM_mereology.xml` = `0` / `24`), read only under
`<mereologyRaise>` (unset ⇒ no bound ⇒ byte-identical). **Refine for non-text models**, where size
is the committed-property count in the relevant (e.g. char-class or frequency) basis.

## Implementation plan (sequenced, 2026-06-16)

The consolidated roadmap, folding in this session's refinements. Status: ✓ done · ◑ partial ·
→ next · ✗ deferred. Each step is gated behind `<mereologyRaise>` and byte-identical with the
flag off unless noted.

**Workstream A — Order-0 mereology: the support machinery + binding (the `.where` tower).**
- **A1 ✓** `RunStructureLayer` measure — runs / gaps / `contained_mask` / extent over a span set,
  fixed-shape, owned by WholeSpace, called gated+read-only in `_stage0_unity_forward`. *(committed)*
- **A2 ✓ (S3)** Word-whole binding — `_embed_radix` stashes `word_groups`; `_autobind_word_wholes`
  binds a token's spell-out pids to ONE shared whole (keyed by text), `maybe_raise_order` fires.
  *(uncommitted; suite 2518/0)*
- **A3 ◑ (S4)** `tightest_container` — force-#1 (largest part ↔ smallest whole) from
  `contained_mask`. The "A isa B" edge, computed; **not yet recorded into the taxonomy**.
- **A4 → (S4-bind)** Record the isa-edges as meronomy edges, under the **generic Universe /
  "sentence" TYPE** (minted once — no churn). Trivial on flat word-spans; becomes non-trivial once
  B (property-tiling) yields nested/multi-class tilings. Depends on **B**.
- **A5 ✗ (S5)** Higher-order part synthesis — replace the raise's mean-combine with
  `SigmaLayer.synthesize_over_set` (needs a *trained* σ at SS width — not a trivial swap) + prune /
  relink the moot per-part edges WITHOUT orphan churn; plus force-#3 (singleton → analyse or drop).
  Invertibility follows contiguity (see "Ramsification with the towers"): contiguous ⇒ synthesize
  (invertible via `SigmaLayer.generate` balanced split); discontiguous ⇒ categorize a common code
  (lossy). High-order codes reserve **contiguous `.what` blocks** and a **per-order symbol table**.
- **A6 ✗** Overcomplete tiling bounded by the basic level (see "Overcomplete tiling bounded by the
  basic level"). The part tower matches the input against `subspace.what` and returns MANY parts
  (not a partition), recombined by IS paint/average, keeping only matches whose `.what`-size (count
  of committed `±1` properties; for text = letter count) lies in `[basicLevelMinSize,
  basicLevelMaxSize]` — config present (`architecture.basicLevel{Min,Max}Size`, text = `0`/`24`),
  matching/pruning to be wired (the over-collection lifecycle prunes the redundant matches).

**Workstream B — Analysis = property-tiling (the π chunker), one unified property-token API (S6).**
- **B1 →** Char-class property backend — `letters` / `digits` / `whitespace` / `punctuation` as
  content-keyed masks (fixed-shape byte-range tensor ops); the run→span cut in the eager stem,
  generalizing `stage_analysis_spans`' whitespace special case. Each = a binary tiling → `.what`
  (class) + `.where` (regions). Wire into `Codebook.materialize_property`'s documented seam
  (approach (a): unify, don't fork).
- **B2 →** OR-union over `.index` — combine multiple selected property rows into one tiling
  (`letters ∪ digits` vs complement); the analysis-side OR, dual to PartSpace's AND-of-particles.
- **B3 ✗** Global / background properties — the low-frequency whole-ranging characterization
  (image frequencies); the **no-localized-`.where`** (abstract, Principle-4) case; reuse the
  sinusoidal `materialize_property` basis + `descriptor_roles` (`ROLE_LF_COARSE`).
- **B unblocks A4** (real containment for `contained_mask` / `tightest_container`).

**Workstream C — Ratio-driven refinement + three-aspect contiguity routing (S8).** CS reads the
`.where` run-structure per representation: **contiguous → refine** (more σ/π in that `.where`),
**discontiguous → raise order**, **zero → null**; integrate (→PS σ) vs disintegrate (→WS π) by
run-count. Wrong part/whole ratio requests further synthesis/analysis until the granularity is
right (the convergence loop). Depends on A+B.

**Workstream D — Radix-climb σ + the real MM_20M fix (S7).** End-to-end on `data/MM_mereology.xml`:
let σ climb (radix promotion) and π descend (property tiling) until word-granularity parts emerge
and the ratio is right. NEVER mutate `MM_20M.xml`. Depends on B+C.

**Workstream E — Types + support (cross-cutting frame).** Both towers are over TYPES; mereological
support (extensional token-set / intensional region) is the distinguishing axis. "Many tokens →
type" is the SAME raise as "bytes → word". Wire order tracking (`abstraction_order` /
`record_fold` live) + the **footprint** (support set) so abstract (no-`.where`) types keep their
meronymy. Higher-order parthood = footprint-`.where`-containment (A+D) + provenance (`part_chain`)
+ optionally lattice absorption / `synthesize_over_set` for novel codes.

**Workstream F — Taxonomic subsumption (the SEPARATE symbolic/relational machine).** `cat ⊑ animal`
lives in `RelativeTruthStore` (the relative-truth table = CS symbolic taxonomy), learned from
**trusted language** + category-based **induction over examples**, with **π meet-absorption** on
property codes as a *similarity weight* — NOT `.where`. Two gates decide where a user truth lands:
**reducibility** (encodable in the codebook via grammatical operations, else stored as an IDEA) and
**trust** (tetralemma `(t,f,b,n)`); codebooks are assumed valid (no per-proposition trust). Cited
cognitive grounding: subsumption is symbolic/inferential, not perceptual (Rosch / Markman / Waxman
& Gelman / Politzer / Murphy & Medin), a developmental hybrid.

**Workstream G — Memories (three tiers + episodic LTM).** Codebooks (codes, assumed valid) · CS
meronymy + symbolic taxonomy (relations = the truth tables, trust-bearing, only for the
reducible-and-trusted portion) · IDEAS / relations-between-ideas (STM transient, LTM persisted —
both trust-bearing). The **episodic LTM** (persisted CS `.events` / PS `.what`+`.where`+`.when`,
distinct from the existing discourse-AR "LTM") verifies a general relation by retrieving order-0
instances and checking parthood where `.where`/co-occurrence still hold → feeds the TruthLayer
trust. **WikiOracle is stateless ⇒ LTM is PROVISIONED, not accumulated: ✗ FUTURE TODO — pre-parse
Wikipedia into ideas (absolute) / relations (relative) + per-memory trust and load it as LTM.**

**Critical path:** ~~B1 → B2 → A4 → C → A5 → D~~. **STATUS (2026-06-18):** **B1 ✓, B2 ✓** (char-class
tiling + OR-union); **A4 ✓** (cross-tower `.where`-containment edges via `record_cross_tower_meronomy`,
live in `_autobind_cross_tower`); **C ◑** (the over-collection lifecycle + σ-synthesis are built and
live-wired — `refine_over_collected`/`synthesize_higher_order`; the π-analyse split is a deferred
nice-to-have, its property-source `factorize_over_set` built); serial dual view (2a/2b/2c) +
word/object/meta (A/B/C) + the §6c context prelude all landed. **The cleanly bridge-independent,
relation-level work is essentially complete.** The REMAINING items all need the
**symbolic↔subsymbolic / captured↔host bridge** or are new subsystems: **A3** (consume the
compiled-forward `tightest_container`/`route_hint` into the taxonomy — captured↔host), **A5** (the
tower-codebook σ-over-set geometric realization + prune/relink — needs a trained σ), the **π-split**
(factorize→property→split, captured↔host), **D** (the radix-climb MM_20M mean-collapse fix —
end-to-end, depends on everything). **F/G ✓ (authority→ideas truth encoding + reasoning + STM→LTM
trust) BUILT 2026-06-18** as the gated `<truthIdeas>` Truth/Ideas subsystem (stages 1–5; see the
"Handoff build stages" list below): reduce-or-describe routing into the WS-META / RelativeTruthStore
two homes, scalar trust collapse, STM→LTM trust persistence, the `reason` modus-ponens engine, and the
`verify_relation` mechanism. The one remaining F/G piece is the episodic SOURCE — provisioning a
pre-parsed Wikipedia / `.events` exemplar LTM (Workstream G FUTURE TODO, no owner). A3 / A5 / π-split /
D still warrant their own focused builds.

> **Substrate (partly built).** `Codebook.property_basis` + `Codebook.materialize_property(index,
> n_positions)` ([Spaces.py:2081](../../bin/Spaces.py)) already read rows as PROPERTIES
> (whole-ranging) and tile the input via a positional basis: `>0` has-property, `≤0` complement
> (the sinusoidal worked example). **Unwired (= S6 property-tiling π):** (a) the CHAR-CLASS
> properties (letters / digits / whitespace / punctuation) as the content-keyed backend (the
> documented "precomputed span mask" seam); (b) the **OR-union over multiple `.index`-selected
> properties** (today `materialize_property` returns the regions per-property, not unioned). The
> whitespace cut in `stage_analysis_spans` is the one hard-wired special case of this. Wiring
> these unblocks `contained_mask` / `tightest_container` on REAL nested/multi-class tilings and
> is the documented MM_20M-fix mechanism (π descends from the Universe by tiling until the
> part/whole ratio is right).

### Build consequence

The `.where`-containment order-raising binding (Stages S3/S4 below) builds **order-0
MEREOLOGY** (byte ⊑ word word-formation; the dormant `maybe_raise_order` fires on a multi-part
word). **Taxonomic subsumption is a separate, symbolic/relational machine** (relation store +
trusted language + reasoning over examples + episodic-LTM verification), not a `.where` binding.

## The CS symbol table + taxonomy — relation-only; data model + lifecycle (2026-06-17, Alec)

Resolves *where* the meronomy lives and *how symbols behave dynamically*.

### Ownership and access
- **ConceptualSpace owns two things:** (1) the **symbol table** (cross-tower part↔whole
  relations), and (2) the **taxonomy** over those symbols. This is the spec's "the corpus
  callosum, in ConceptualSpace, builds the single meronomy out of the two towers."
- CS **reads** the PartSpace part codebook and the WholeSpace whole codebook (read-only
  lookups) and **writes only its own** symbol table + taxonomy. **No space mutates another's
  codebook.**
- **Access rule:** per-forward *state* (events, live `.where`, which parts/wholes appeared)
  flows **only via subspaces passed into `forward`/`reverse`** — never peer-reads of a peer's
  transient `_forward_input` / `_embedded_input` / `subspace.where`. Persistent **codebook
  reads** are the sanctioned cross-space access for CS (the meronomy owner).
- **Preallocate** the table/position capacity so binding is index-assignment → it runs **in
  `forward`**, not the `Reset` growth-workaround.

### Symbols are RELATION-ONLY
- A symbol has **no learnable vector and no codebook row** — it is a *relation* tying PS
  part-codes ↔ WS whole-codes (read from their codebooks). There is **no symbol codebook**
  (the current `insert_symbol` meta-vector seeding is retired).
- **Per location (`.where` region):** a symbol ties *the parts covering that location* to
  *the wholes covering that location* — the "space in between" part-codes and whole-codes.
- A symbol is defined by **two INDEPENDENT multi-valued attribute-sets** keyed by the symbol:
  **`Parts(S)`** (the part-codes / sub-symbols on its part side) and **`Wholes(S)`** (the
  whole-codes / super-symbols on its whole side). Independent — *not* a global
  `X∈Parts(S) ⟺ S∈Wholes(X)` dual — because a reified relation's operands aren't the operands'
  meronymic parts/wholes. Reverse lookups are derived on demand.
- A **relation between symbols is reified as a new symbol**: `C : A <= B` ⇒ `Parts(C)={A}`,
  `Wholes(C)={B}` ("the cat-object is a cat-word"; no larger whole ties them). The word↔object
  META is this special case.

```
A : cat-object <= cat-object-properties    Parts(A)={cat-object, felix-object}
B : cat-word   <= cat-word-properties       Wholes(A)={cat-object-properties}
C : A <= B  (the object is a word)          Parts(C)={A}   Wholes(C)={B}
```

### Word / object / meta creation (BUILT 2026-06-17, Alec)
> *"PS and WS are constrained to give word-parts and word-wholes. CS can create the word-symbol (A)
> and object-symbol (B) and their meta (C). B will have as parts and wholes only atoms and the
> universe, but these will be successively refined."*

`ConceptualSpace.create_word_object_meta(word_parts, word_whole, key=None)` ([Spaces.py](../../bin/Spaces.py))
mints the three relation-only symbols per perceived word:
- **A = word-symbol** — `Parts(A)` = the word-parts (PS part-codes), `Wholes(A)` = the word-whole
  (WS whole-code). The orthographic word; A accumulates word-parts across presentations.
- **B = object-symbol** — `Parts(B) = {ATOM}`, `Wholes(B) = {UNIVERSE}` **initially**: the referent is
  maximally unspecified (bottom pole ↔ top pole) and is **successively refined** (σ synthesizes the
  atoms into higher-order parts; π splits the universe into finer wholes — the lifecycle loop).
- **C = meta** — `reify_relation(A, B)`: `Parts(C)={('sym',A)}`, `Wholes(C)={('sym',B)}` — the
  word≡object binding (the second-order MetaSymbol above).

`ATOM` / `UNIVERSE` are key-only lattice-pole sentinels ([Layers.py](../../bin/Layers.py), beside
`WORD`), references in the relation-only table — **no vector, no codebook row**. **Refinement guard:**
`resolve_identities()` does **not** collapse a pole-pair (`ATOM`/`UNIVERSE`) — B has the *shape* of a
1:1 identity but is the unspecified placeholder, so it stays in `symbols_needing_processing()` until
the lifecycle specializes it to a tie between **concrete** codes. Idempotent per surface `key`. Tests:
`test/test_cs_symbol_table.py` (`create_word_object_meta_*`, `resolve_identities_does_not_collapse_
unspecified_object`).

**LIVE-WIRED (2026-06-17, gated `<mereologyRaise>`).** `_autobind_word_wholes` now **returns** its
per-word descriptors `(word_parts, word_whole, surface_key)` (it still touches only the WS handle, so
the `self=None` standalone tests are unchanged); the orchestrator `_maybe_autobind_meta` then calls
`create_word_object_meta` for each, minting A/B/C alongside the legacy WS word-whole binding and the
cross-tower `.where` edges. Additive CS state, dark by default (byte-identical flag-off). Test:
`test/test_mereology_word_binding.py::test_gate_on_creates_word_object_meta`.

**How B (the object) is refined — AUTHORITY, not the perceptual lifecycle (Alec, 2026-06-17).**
> *"A valid cognizer must learn the object through experience, reasoning, or authority. We are a text
> interface right now, so authority is the primary means (the TruthSet from the user). Reasoning
> happens later, after we create a working knower."*

So B's `ATOM`/`UNIVERSE` poles are **not** refined by the σ/π over-collection lifecycle (that refines
the *perceptual* part/whole structure of the towers). The OBJECT is refined by **authority** — the
user's **TruthSet** (trusted English sentences). **Experience** (episodic LTM, Workstream G) and
**reasoning** (category-based induction over examples) come **later**, once there is a working knower.

**The truth is mostly NOT a codebook part-whole edge — it is a relation between composed IDEAS
(Alec, 2026-06-17).** A copular "is-of-predication" sentence reduces to a part-whole relation over the
two codebooks **only when both sides are single codes**. "*bowling pins are white*" does **not**:
"bowling pins" is a syntactically-composed NP, not one code. So the general case must be encoded as a
**relation between syntactically-composed IDEAS in STM/LTM** — structures that **need not be codebook
codes** and are **often more complex than the isomorphically-defined `_sym_*` symbols**. This is the
spec's **reducibility gate**: only the *reducible single-code* subset becomes a `_sym_*` part-whole
edge (or refines B's poles directly); everything else is an **idea / relation-between-ideas** in the
STM/LTM truth memory (`RelativeTruthStore` / `TruthLayer`; Workstreams **F + G**). Build consequence:
the authority channel targets the **ideas memory** (composed-idea relations), **not** a naïve "route
every assertion into `_sym_*`" — that only works for single-code subjects.

**Reducibility is gated by TRUST (Alec, 2026-06-17).** Even a single-code-reducible truth must **not**
be written straight into the codebook part-whole table — the codebook is **assumed-valid / absolute**
(no per-proposition trust), so promoting user **testimony** (3rd-person authority) into it would
**treat testimony as direct experience**. Keep the reduction **scoped by the trust score** (tetralemma
`(t,f,b,n)`): testimony lives in the **relative / trust-bearing** store; it does not become an absolute
codebook edge merely because it is structurally reducible. Reducibility AND trust are **both** gates.

**The truth/ideas subsystem is the OUTPUT of serial parsing**, not a separate prerequisite: serially
parsing a sentence and storing it as an absolute or relative truth IS the subsystem — so it falls out
of the serial-mode work below, which the build heads toward anyway.

### Serial-mode word-at-a-time loop — strategy (mapped 2026-06-17, Alec deciding)
Alec: in serial mode FORCE PS to combine letters up to word-sized chunks and FORCE WS to isolate
"words", LOOP over PS+WS one word at a time so each word adds its object (B) + meta (C) *before*
syntax, and "depending on our strategy, we can drop the gaussian masking." A 5-facet read-only map
established the **current state**:
- **A slot ≠ a word.** Radix `spell_out` emits a RUN of letter-chunk percepts; `InputSpace.word_at(p)`
  ([Spaces.py:8376](../../bin/Spaces.py)) ticks once per *chunk* until frequency-promotion
  ([Layers.py:9858](../../bin/Layers.py), default 4×) collapses a word to one percept. The word
  boundary is known in `_embed_radix` (`word_groups`, [Spaces.py:9749](../../bin/Spaces.py)) but is
  **not** used to force one-pid-per-token; PS explicitly declines a whitespace force.
- **The per-word body is torch.compile-captured** (`_per_word_body_step`, [Models.py:6671](../../bin/Models.py);
  loop at 6978) — which is *why* the A/B/C mint was exiled to eager `Reset`. A per-word B/C hook must
  run eager / `@torch.compiler.disable`.
- **A/B/C fires at the sentence boundary, AFTER compose** (`_maybe_autobind_meta` ← `Reset`), so today
  it can only inform the *next* sentence — wrong place to be "the basis for syntactic analysis."
- **Word↔object substitution primitives** `resolve_ps_terminal` ([Spaces.py:15229](../../bin/Spaces.py))
  + `ReferenceTable` ([Spaces.py:17509](../../bin/Spaces.py)) exist with **zero production callers**
  (the spec's ✗ verify/wire item).
- **Gaussian "mask"** is a per-word attentional WINDOW (`gaussian_window_word`,
  [Models.py:2447](../../bin/Models.py)), not BERT masking; it supplies neighbour context. D3 loss
  reconstructs the UNMASKED input regardless.

**Strategy options** (full analysis in the workflow transcript):
- **Option A — reuse the existing serial body + eager per-word B/C shim + word→object substitution
  around compose.** Force PS via a runtime `_embed_radix` "one pid per lexer token" branch; KEEP the
  Gaussian window (it now centres on a real word); new `serialObjectMeta` gate (independent of
  `<mereologyRaise>`); flag-off byte-identical. Medium blast radius; **closes the ✗ substitution item**.
- **Option B — relocate the word loop into ConceptualSpace** (loop + mint in one eager object; WS span
  carrier drives each step). The architecturally "right" home, **drops** the Gaussian window (WS whole
  + recurrent C→P/S carry subsume context), but **moves the torch.compile capture boundary** → largest
  re-pin.
- **Option C — config-forced chunking + minimal per-word meta read via the existing
  `_build_category_context` bridge, no substitution.** Smallest, fast probe; does **not** fully satisfy
  "B/C as the basis for syntax."

**DECISION (Alec, 2026-06-17): Option A, refined — both PS and WS process the ACTIVE WORD ONLY (hard
mask).** During the per-word step the Word-symbol A is built from spatially-correct parts and wholes:
- **PS parts:** *no part with a `.where` outside the active word* (parts ⊆ word span);
- **WS wholes:** *no whole that fails to include the entirety of the word in its `.where`* (whole ⊇ word span).

This is a **HARD mask** to the active word (it replaces — does **not** keep — the soft Gaussian window).
**Acknowledged downside:** hard masking forgoes the **word2vec-like contextual embedding** the Gaussian
window provided (no neighbour-context shaping of a word's code). Accepted for now.

**Object-as-vector (B is relation-only):** the word's own vector stands in for the object until B is
**grounded by authority** (the TruthSet); the word→object substitution is a no-op until the object is
known, then becomes live. Other settled defaults: B/C gets its **own gate** (`serialObjectMeta`,
independent of `<mereologyRaise>`); flag-off byte-identical.

**Context for the hard mask = the §6c sentence-protocol prelude (DEFAULT CUTOVER, 2026-06-18, Alec).**
The hard mask drops neighbour context (no word2vec-like embedding). That context re-enters via the
**initial subsymbolic-order pass** — `_sentence_prelude` ([Models.py:6651](../../bin/Models.py)) runs
`subsymbolicOrder` **whole-sentence** pumps (both towers, EMA on) → a **gist** → `set_intent` priming
both towers (intent-only: feedback restored, no STM push; eager, before the captured per-word loop). So
a hard-masked word is processed **against gist-primed towers** — sentence context conditions its
parts/wholes through the **intent channel**. **Cutover:** `sentence_protocol` default changed
`False → (symbolicOrder ≥ 1)` ([Models.py ~639](../../bin/Models.py)) — **ON in serial, OFF in
parallel** (the prelude is only invoked from `_forward_body_per_word`); explicit `<sentenceProtocol>`
overrides. Triage of the ~18 serial configs: **only** `test_sentence_protocol::test_protocol_off_by_
default_and_dark` failed (it pinned the old default) — updated to the new contract (on-in-serial /
dark-when-forced-off / off-in-parallel); all serial configs' functional tests stayed green.

**Build increments (incremental, suite-green per step):**
1. **✓ DONE (2026-06-17).** Serial mereology config — `data/MM_mereology_serial.xml` = MM_mereology +
   `symbolicOrder=1`. The serial+radix path (previously unexercised) **constructs and forwards
   cleanly**; `terminalConceptualSpace_ref` wiring verified. Suite 2564/0.
2. **Word-span hard mask + per-word commit (mapped + adversarially verified 2026-06-17).** Replace
   `gaussian_window_word` with `word_span_window` (the HARD same-word mask: sum the slots sharing
   slot `k`'s word index → the active word's rep `[B,1,D]`), gated `serialObjectMeta`, byte-identical
   off. **Verified correction:** `MM_mereology_serial.xml` is `synthesis=radix`, so `_embed_radix`
   emits **one slot per pid** — an unfamiliar word spans MULTIPLE byte-pid slots (grouped by
   `word_group_grid`); promoted words are 1 slot. So a **per-word commit gate** (fire the STM push +
   the eager A/B/C **once per word**, at the last slot of each word) is *genuinely required* (not a
   no-op). Byte mode has no `word_groups` → `word_span_window` degrades to the single-slot fallback.
   - **2a ✓ DONE:** `word_span_window` pure helper (`Models.py`, after `gaussian_window_word`; reads
     none of `self`, fully unit-tested in `test/test_serial_object_meta.py`, 6 tests). Unwired ⇒
     byte-identical.
   - **2b ✓ DONE (2026-06-18, captured-loop wiring).** `serialObjectMeta` config + XSD; stamped on the
     model **+ InputSpace** (reaches `finalize_stem`) **+ CS** (a NEW stamp site — read live in
     `_create_per_stage`, order-safe, since `_mereology_raise` never reaches CS). `InputSpace.finalize_stem`
     surfaces `word_group_grid` as the capturable `[B,N]` `_word_index_N` (+ `_word_last_slot_mask` =
     last-slot-of-word AND active), flag-guarded (None elsewhere ⇒ consumers fall back to per-slot).
     `_per_word_body_step` branches `gaussian → word_span_window` at the window call (PS processes the
     active-word block) and gates the STM push on `commit_b_1` (per-word last-slot) so a radix
     multi-slot word pushes ONE idea. **Capture-safe** (compiled forward + capture-gate test pass;
     the branch is a const-folded Python bool, the mask/slice are fixed-shape). **Byte-identical off**
     (flag-off → gaussian + per-slot, `_word_index_N` not built). Tests: `test_serial_object_meta.py`
     (6 unit + 3 integration: stamps/tensors, commit-once-per-word, flag-off-no-build). The A/B/C mint
     **already fires** under the config's `mereologyRaise` gate (no gate-widen needed here). **Host
     STM-depth mirror** stays per-iteration (conservative upper bound → at worst an early-but-safe
     reduce in long radix words; raise `stmCapacity` if needed). **2c ✓ DONE (2026-06-18, the WS half of the
     dual view — via the prelude, not a per-word unity).** Alec: *"pass both IS and CS to both PS and
     WS… PS/WS look at INPUT for the first step, then process subsymbolically, then symbolically."*
     The clean realization: the §6c **prelude** IS the subsymbolic pass, and its pump 0 has the EMPTY
     CS seed — so feeding WS the sentence **unity** (`_staged_concepts_in`) at pump 0 routes through
     `_stage0_unity_forward` (the legal stage-0 path; **no** repeated-injection NotImplementedError,
     and **no** per-word byte-masking needed). `_sentence_prelude` now does
     `ws.forward(prevCS, IS_concepts=_staged_concepts_in if (serial_object_meta and pump==0) else None)`
     — so both PS (input/atoms) and WS (unity) see the input at step 0, then subsymbolic pumps (CS),
     then the per-word **symbolic** loop. Gated `serialObjectMeta` (other serial configs' prelude is
     byte-identical: `IS_concepts=None`). Verified: WS runs `_stage0_unity_forward` once with the real
     unity at pump 0. Tests: `test_serial_object_meta.py::test_ss_analyzes_unity_at_prelude_pump_zero`
     (+ gated-off). The per-pid stack vs per-word STM reconstruction test remains a nice-to-have.
3. **Per-word A/B/C before compose** — the eager mint **already** fires once-per-word keyed by surface
   text at `ConceptualSpace.Reset`→`_maybe_autobind_meta`→`create_word_object_meta` (a per-`p` eager
   seam inside the compiled forward is impossible; the `@torch.compiler.disable` idea was wrong — a
   disabled call is itself a fullgraph break). So step 3 ≈ widening that gate (part of 2b).
4. **Word→object substitution** — wire `resolve_ps_terminal` / `ReferenceTable` around the per-word
   compose (no-op until B is authority-grounded); inverse on the reverse path.
Workstream B/C lifecycle and the authority→ideas truth encoding ride on top once the per-word loop
delivers clean word-symbols.

### Lifecycle — over-collection triggers refinement; convergence → identity
- **Over-collection is ACTIONABLE.** Too many parts OR too many wholes on a symbol triggers
  refinement:
  - **too few parts per symbol → SYNTHESIZE (σ):** group parts so a symbol covers *more* parts
    (a higher-order part);
  - **a whole over-subscribed by too many symbols → ANALYSE (π):** split that whole into finer
    wholes (*fewer* wholes per symbol).
  - **First round:** PS = atoms, WS = universe. The universe is on **too many** symbols, and
    each part is on **only one** symbol ⇒ *both* remedies fire (more parts per symbol via σ;
    split the universe via π). **The symbol table drives both towers + order-raising.**
- **Triggering symbols are TRANSIENT:** a symbol that signals "restructure here" is **retired**
  once it has triggered the analysis / synthesis / order-raise (the restructuring supersedes
  it).
- **Convergence → IDENTITY (id of indiscernibles):** when a symbol's sets collapse to exactly
  **one part + one whole**, the multi-valued sets **disappear** (their role is subsumed by the
  codebooks) and the symbol becomes a stable **identity tie** between that part-code and that
  whole-code — σ-up meets π-down at the object (Principle 1). Non-triggering symbols **persist**.

> **What the over-collection step actually drives (2026-06-17, Alec).** The per-code refinement
> already happens in the **existing subsymbolic loop** (it treats the codes appropriately over
> `subsymbolicOrder` iterations). So the **built** deliverable here is just to **zero out the 1:1
> mappings** — a 1:1 symbol is the resolved identity and needs no further processing
> (`ConceptualSpace.resolve_identities`; the still-active set is `symbols_needing_processing`).
> **✗ Second step (doc-note, not built):** send back **only** the symbols that are **N:1 or 1:N for
> large N** — too many parts for one whole, or too many wholes for one part — by handing that
> symbol's **parts back to PartSpace** (for finer synthesis) or its **wholes back to WholeSpace**
> (for finer analysis). A large N:1 or 1:N indicates a *lack of flexibility* at that symbol; the
> send-back is what requests the extra σ/π that restores it. (This is the explicit feedback that
> complements the subsymbolic loop's implicit per-code processing.)

### The analytical basis (the current gap)
- The right WS analytical basis is **binary properties** — analysis = **OR over properties**
  (0-fill elsewhere); currently insufficient. The property basis is the S6 build (partly done:
  `char_class_region` / the `materialize_property` content-keyed seam / OR-over-`.index`).
- Meanwhile, with WS codebook codes available, each input is **tiled with codes over `.where`**,
  and parts↔wholes are related mereologically by that `.where` (cross-tower `.where`-containment).

### How the built primitives realize this
| Built | Role in the model |
|---|---|
| property-tiling (`char_class_region`, `materialize_property` seam, OR-over-`.index`) | the binary-property analytical basis |
| `record_cross_tower_meronomy` (PS `.where` ⊆ WS `.where`) | the `.where` relating that mints a symbol per location |
| `RunStructureLayer` `n_runs` / `route_hint` / `tightest_container` | the over-collection / part-whole-ratio trigger (refine / raise / null; tightest = smallest whole) |
| `maybe_raise_order` | the σ-synthesis / order-increase |
| lattice `taxonomy_parents` / `ps_children_of_whole` | the multi-valued `Wholes(S)` / `Parts(S)` views |

**To build (new dynamics on top):** ~~the **retire-on-trigger** lifecycle; the **1:1 → identity
collapse**; the trigger driving **both** π-analysis and σ-synthesis~~ — **✓ DONE + LIVE-WIRED
(2026-06-18, Workstream C):** `ConceptualSpace.refine_over_collected(k_parts, k_wholes)`
([Spaces.py](../../bin/Spaces.py)) is the over-collection lifecycle pass: resolves 1:1 identities
first (excluded), then for each over-collected symbol **APPLIES σ-synthesis** — too many PARTS →
`synthesize_higher_order(parts)` mints a HIGHER-ORDER symbol H grouping them (`Parts(H)`=provenance,
the relation-level analogue of `part_chain`/`maybe_raise_order`; tagged **raised** so it is never
re-refined), request carries `'result': H` — and **retires the trigger** (retire-on-trigger). Too
many WHOLES → `op='analyse'` request is **emitted but the π-split is deferred** (the finer-whole
criterion is underspecified). **Live-wired:** called in `_autobind_cross_tower` right after
`resolve_identities` (gated `<mereologyRaise>`, host-side at Reset; additive). `symbols_needing_
processing` excludes raised symbols. Idempotent. Tests: `test_cs_symbol_table.py`
(`refine_over_collected_*`, `synthesize_higher_order_*` — 7).

**π-analyse split — property-source BUILT (2026-06-18, Alec's criterion).** Alec: *"an over-subscribed
whole should split by ADDING A PROPERTY; in subsymbolic mode by FACTORING the PiLayer output (criterion
open); with numerous properties, a SOFT SUPERPOSITION lets the data decide."* The property-source —
`PiLayer.factorize_over_set(y, M)` ([Layers.py](../../bin/Layers.py)) — is now built: the **log-domain
dual of `SigmaLayer.generate`'s balanced split** (recover the summed log-membership `lx = (log(_to_mult
(y))-b)@W⁻¹` as `reverse` does, split equally by M, exit — where σ splits a SUM by M in the atanh
domain, π splits a PRODUCT by M in the log-mult domain). `M=1` reduces to `reverse`; folding M equal
factors back recovers `y` (round-trip tested, `TestPiLayer`). Factoring an over-subscribed whole's code
exposes its constituent **properties** to split on.

> **NICE-TO-HAVE (deferred 2026-06-18, Alec — "unneeded complexity for the moment").** The full
> **π-analyse split** — wiring `factorize_over_set` → the splitting property → the relation-level split
> of the over-subscribed whole — is **deferred**. The `op='analyse'` request is still emitted by
> `refine_over_collected` (detect-only); applying it (especially the "few-properties" general case,
> with the partition criterion + soft-superposition over candidates) is a future refinement. The σ
> half (synthesis) is the live one; the property-source (`factorize_over_set`) is built and tested for
> when the split is wired. Likewise the σ tower-codebook geometric realization
> (`SigmaLayer.synthesize_over_set`, currently relation-level provenance only) is a nice-to-have.

### Relocation (WholeSpace → ConceptualSpace) — relation-only ⇒ mainly movement
Because symbols are relation-only, **there is no codebook split**: move the *relational* tables
+ methods from WholeSpace to ConceptualSpace and **drop the meta-vector creation**
(`insert_symbol` for metas).
- **Moves:** the taxonomy / `Parts`-`Wholes` dicts (+ position maps, idempotency cache,
  `meta_trust`, `part_chain`, LBG accumulators) and the ~25 methods; the autobind (already on
  CS) switches `ws.method()` → `self.method()`; persistence (`vocab_extras` / `load_vocab_extras`
  + `Models._collect/_restore_vocab_extras` + a one-generation checkpoint shim +
  `_migrate_signed_int_taxonomy`); Models wiring (CS built before WS; CS keeps read refs to PS +
  WS codebooks); repoint the ~18 test files.
- **Staged, suite-green per stage:** Stage 1 tables+methods on CS with WS delegation shims (suite
  stays green) → Stage 2 repoint autobind + Models wiring → Stage 3 persistence + checkpoint shim
  → Stage 4 repoint tests, drop shims.

### Migration reality check (2026-06-17, verified against source — corrects the framing above)
A read-only blast-radius mapping (6 facets) + adversarial critique, **verified against the live
tree**, found the "relation-only ⇒ mainly movement / no codebook split" framing is **wrong on the
points that matter**. The corrected, code-grounded picture:

1. **`insert_symbol`/`insert_meta` are CODEBOOK-COUPLED — they cannot just move to CS.**
   `insert_symbol` hard-requires `self.subspace.what` be a `Codebook` ([Spaces.py:14844](../../bin/Spaces.py),
   raises otherwise) and mints rows through the **shared `_paired_next_row` / `_paired_orth_to_sem`
   cursor** seeded from `well_known_atoms` (shared with the legacy `insert_paired_word` lexicon).
   `ConceptualSpace.subspace.what` is a **plain Tensor**, not a Codebook. So the **Codebook + the
   atomic row/position allocation stay on WholeSpace** (the owner); CS gets **only the pure-dict
   taxonomy bookkeeping**.
2. **The META taxonomy is TERMINAL-SS-scoped, not per-stage.** `_maybe_autobind_meta` targets
   `terminalSymbolicSpace_ref` ([Spaces.py:12509](../../bin/Spaces.py); docstring: *"the META
   taxonomy is owned by the canonical (terminal) SS … growing a per-stage SS codebook would overrun
   the where-space registry"*). There are **N per-stage CS instances but ONE terminal table** — a
   naive `ws.`→`self.` flip would **fragment the shared taxonomy into N**. The migration must target
   a **single terminal CS**; do NOT pair each per-stage CS with its co-stage WS.
3. **`ConceptualSpace(Space)` ≠ `WholeSpace(PerceptualSpace)`.** `_peer_percept_store` /
   `insert_percept` ([Spaces.py:14790/14803](../../bin/Spaces.py)) are PerceptualSpace machinery CS
   lacks — the percept-seed/allocator helpers stay on WS.
4. **Corrected decomposition.** CS owns the pure-dict taxonomy (`taxonomy`, `taxonomy_parent_map`,
   `meta_pair_to_idx`, `meta_trust`, `part_chain`). WS keeps the `Codebook` + a new **atomic
   `allocate_symbol_row(init_vec) → (pos, row)`** that mints the row AND binds `pos↔row` in one
   owner; CS calls it and records the position. **One owner for `_pos_kind` / `_ws_pos_to_row` /
   `_ws_row_to_pos`** (the `"ws"` vs `"meta"` tag desyncs if split) — keep them **with the allocator
   on WS**, or move the allocator too; **never split allocator from the maps it mutates**.
5. **Persistence is wider than the taxonomy.** `vocab_extras` also serializes `well_known_atoms` /
   `_paired_orth_to_sem` / `_paired_next_row` ([Spaces.py:15917](../../bin/Spaces.py)) and Models
   reads `well_known_atoms` off WS ([Models.py:1329/1573](../../bin/Models.py)). Either keep
   `vocab_extras` on WS (serializing a CS taxonomy sub-blob) or add those three to the inventory.
6. **RadixLayer.reverse repoint site is [Models.py:1742](../../bin/Models.py)** (the caller passing
   `symbolic_space=ws`), not the layer body — `RadixLayer` holds no CS handle.
7. **vector→relation retirements are DEFERRED past S3** (need their own design + sign-off): the
   `_nearest_symbol_target` MSE quantization loss (a gradient-path training signal with no row to
   regress to under relation-only), `_snap_content` decode-by-similarity, and `RelativeTruthStore`
   cosine → graded collection-overlap + component-wise ternary negation (reuse `NegationLayer`).
8. **Verified baseline = 2557 passed / 0 failed (2641 collected), 2026-06-17** — *not* the stale
   2505 some memories cite. Re-pin the per-stage suite-green gate to this.
9. **Scope trim:** `test_search_then_mint.py` / `test_serial_stm_split.py` touch no migrated method
   (grep-confirmed; they use the separate `ReferenceTable`) — out of scope. ~9 WS-relevant test
   files, not ~18.

**Verdict:** the migration is **not "mainly movement"**. Two relocation readings were surfaced and
**Alec chose RELATION-ONLY COMPLETION (2026-06-17)** — *not* a codebook move:

> CS becomes the **sole owner** of the symbol/taxonomy as a **relation-only LUT** (the landed
> `_sym_*` table) that references **existing PartSpace part-codes and WholeSpace whole-codes**
> (whole = a reference to the part). **Symbols stay vectorless** (no codebook row); the legacy WS
> `insert_symbol` meta-vector seed is retired. **WS keeps its whole codebook; PS keeps its part
> codebook.** This matches the "Symbols are RELATION-ONLY" principle above.

Consequences vs. a codebook move: **no new CS codebook**, so **no per-stage fragmentation risk, no
codebook state_dict/VQ/ramsification migration** — only the taxonomy **JSON blob** (`vocab_extras`)
re-homes. The one genuinely hard piece: the legacy meta **vector is load-bearing** (the
`_nearest_symbol_target` MSE quantization loss regresses to it; `SymbolizeLayer.forward` reads
`W[meta_row]` downstream), so **retiring it is a gradient-path change** — sequenced **last**, behind
its own sign-off + behavioral-equivalence harness.

**Staged plan (relation-only completion):**
- **S3a — golden harness (read-only).** Snapshot the META bindings (decode/reverse) + taxonomy
  **AND the codebook rows** (`subspace.what.getW()` for the META/SS rows) over a fixed corpus
  pre-migration. Suite 2557/0. (Review correction: the silent-divergence path is row-mint ordering
  → `_nearest_symbol_target` MSE target, which a decode/taxonomy-only snapshot would miss.) Safe
  under any plan; the behavioral-equivalence oracle for later stages.
- **S3b — CS owns the relation LUT (mechanism revised after adversarial review, 2026-06-17).** The
  review found the original "move the dicts to CS, keep the codebook on WS, forward via `@property`"
  is **internally inconsistent**: `insert_symbol`/`insert_meta` mint a codebook row AND write the
  position dicts **in one atomic call sharing cursors** ([Spaces.py:14786/15433-15439](../../bin/Spaces.py)),
  so splitting dict-ownership from row-mint re-introduces the peer-write coupling. **Adopt Fix #1
  (invert the shim direction):** the legacy taxonomy dicts + position maps + cursors (`_next_position`,
  `well_known_atoms`, `_paired_next_row`) **stay physically on the terminal WholeSpace** (already a
  singleton — `terminalSymbolicSpace_ref`, so there is **no fragmentation to fix**); **CS owns the
  relation-only INTERFACE by reference** — the `_sym_*` table + any new CS relation API forward to
  `terminalSymbolicSpace_ref`. This preserves the atomic row-mint+dict-write, keeps the cursors
  co-located, and needs **zero reader/mutator repointing** ⇒ behavior-equivalent. (True physical
  CS ownership = **Fix #2**: move the dicts + the atomic methods + all three cursors as a unit, WS
  exposing *identity-forwarding* descriptors, not copies — materially larger, deferred.) `_sym_*`
  stays the live relation view over the WS dicts; it is **ephemeral** (not persisted) under Fix #1.
- **S3c — persistence.** Under Fix #1 the physical store stays on WS, so persistence is **unchanged
  / no re-home needed**. (Review correction: `vocab_extras` is **one entangled blob** — taxonomy +
  `well_known_atoms` + `_paired_*` cursor + position cursor + `part_chain`-only-when-non-empty
  ([Spaces.py:15908-15968](../../bin/Spaces.py)); a re-home could not move "only the taxonomy". Fix
  #1 sidesteps this entirely. If Fix #2 is later chosen, the WHOLE `vocab_extras`/`load_vocab_extras`
  pair moves, preserving the inverse-map rebuild and the byte-identical-flag-off `part_chain` rule.)
- **S3d — tests + drop shims.** Repoint the ~9 WS-relevant test files; drop the WS delegation shims.
- **S3e — DEFERRED, separate sign-off (gradient-path change).** Retire the meta-vector seed +
  `_nearest_symbol_target` MSE loss; reroute the `SymbolizeLayer` `W[meta_row]` consumer to a
  relation-derived value; `RelativeTruthStore` cosine → component-wise ternary negation
  (reuse `NegationLayer`). The **only** non-byte-identical, training-signal-changing piece; last.

**Converged conclusion (three verified rounds, 2026-06-17).** Two planning workflows + one adversarial
code review (verdicts: unsafe, unsafe, needs-revision) converge on: **the physical relocation of the
legacy taxonomy is low-value churn** — it is *already* a terminal singleton (no fragmentation to fix),
its methods are atomically codebook-coupled (can't be cleanly split), and the only behavior-equivalent
"CS ownership" is **Fix #1 = ownership-by-reference** (physical state stays on WS; CS is the relation
interface). The **real relation-only capability is S3e** (retire the meta vectors → vectorless symbols
+ component-wise negation), which is a **gradient-path change needing its own sign-off**. The
`_sym_*` relation-only LUT — the thing Alec actually specified — **is already landed** (15 tests).

**LANDED (2026-06-17, Alec: "Do S3 relocation").** Fix #1 executed: `Models` wires a single
`terminalConceptualSpace_ref` (= `conceptualSpaces[-1]`) onto every CS **and** every WS (mirroring the
`terminalSymbolicSpace_ref` fan-out); `ConceptualSpace` gains `_relation_store()` + forwarding relation
read-API accessors (`taxonomy_children` / `taxonomy_parent` / `taxonomy_parents` / `is_meta` /
`ps_children_of_whole`) so **CS is the canonical relation-only owner-by-reference** — callers can
migrate WS→CS behavior-equivalently. Additive, byte-identical (suite 2564/0). Test:
`test/test_mereology_word_binding.py::test_cs_owns_relation_taxonomy_by_reference`.

**S3e is NOT needed — superseded by the A/B/C creation (Alec, 2026-06-17).** "S3e" was a proposed step
to make the *legacy* metas vectorless (retire `insert_meta`'s codebook-row seed + the
`_nearest_symbol_target` MSE loss). But the relation-only symbols **are** the freshly-created A/B/C
(which never carry a vector): A (word-symbol) is built from PS+WS **serial-mode** behavior once
word-parts and word-wholes are ensured, with B (object) and C (meta) minted at the same moment. The
legacy vector-bearing metas are a **separate, coexisting path** — nothing must be "retired" to obtain
the relation-only model. **⇒ S3 is effectively complete** (Fix #1 ownership + A/B/C creation); the
gradient-path change is dropped. Removing the legacy vector machinery later is optional **cleanup**,
not a capability gate.

## Truth / Ideas processing (DESIGN, approved 2026-06-18, Alec — handoff to next session)

How a user's TruthSet (trusted English sentences; authority is primary for a text interface) becomes
stored, reasoned-over truth. This is the OUTPUT of serial parsing (now built: the serial dual view +
A/B/C + the §6c context prelude).

### Map of all truth — the knowledge loci at a glance (2026-06-18, verified against code)

Truth and knowledge live in **eight distinct loci**. The invariant (see "Correctness partition" below):
**ENTITIES** (codes, ideas) are **ABSOLUTE / assumed-valid**; **LINKS** (relations) are **RELATIVE /
trust-bearing**. Testimony writes links, never entities.

| Locus | Holds | Abs / Rel | Trust | Persisted? | Gate | Read by |
|---|---|---|---|---|---|---|
| **User TruthSet** (authority) | trusted English sentences (text + trust) ingested to the absolute corpus | arrives RELATIVE (per-stmt trust), stored ABSOLUTE | scalar DoT ∈ [-1,1] | per-request rebuild — cleared each `store_truths` call ([Models.py:2173](../../bin/Models.py)) | `truthCriterion` (forced 0 during ingest) | `store_truths`; luminosity/falsity loss | 
| **PS + WS codebooks** (ramsified) | assumed-valid prototype codes (PS atoms / WS properties + symbol prototypes); ramsification sidecar records the σ/π fold route → abstraction order | **ABSOLUTE** | **none** (codebook is assumed-valid) | `.W` ✔ in state_dict; ramsification table **not** persisted (rebuilt at build) | `codebook`; ramsification under `mereologyRaise` | forward σ-synthesis / π-analysis |
| **CS symbol table** (knits parts↔wholes; A/B/C) | relation-only symbols tying PS part-codes ↔ WS whole-codes; word/object/meta (A/B/C) | **RELATIVE** (relation-only) | **none** ([Spaces.py:12871](../../bin/Spaces.py)) | transient (host dicts) | `mereologyRaise` | lifecycle (`resolve_identities`/`refine_over_collected`) |
| **Relational hierarchy** (WS META taxonomy + `meta_trust`) | REDUCED relations (predicate=parent / two ideas=children) + autobind percept↔symbol metas | graph ABSOLUTE; relation nodes RELATIVE | **tetralemma** `(t,f,b,n)` ([Spaces.py:14638](../../bin/Spaces.py)) | ✔ `vocab_extras` | `truthCriterion` (learn) + `truthIdeas` (reducible→here) | decode / order-raising; the reduce branch of `reason` |
| **Truth-Ideas: TruthLayer** (absolute corpus) | consistent-proposition corpus; DoT baked into activation magnitude/sign | **ABSOLUTE** | scalar DoT baked into magnitude | buffers ✔ state_dict (but cleared per `store_truths`) | `truthCriterion` | `luminosity`/`ground`/`assess`; truth-modulated loss |
| **Truth-Ideas: RelativeTruthStore** (relations between ideas) | uncollapsed `(np1, vp, np2)` idea triples — the "explicit knowing" / **ineffable** relations | **RELATIVE** | scalar `t−f` (baked into magnitude **and** `_trusts` list) | `np1/vp/np2` buffers ✔ state_dict; **`_trusts` list ✗ (bug — see below)** | `truthIdeas` (ineffable branch) | `reason` / `consequents` / `evaluate` / `verify_relation` |
| **STM** (`ShortTermMemory` + typed STM) | per-presentation idea stack (newest-at-slot-0, cap ≈ 8) + typed metadata (category / order / ref_id) | working IDEAS (entities), not links | **none** | transient (`persistent=False`) | always-on | per-word loop / reduce / intra-predictor |
| **LTM** (`InterSentenceLayer` end-state chain) | per-row deque of persisted STM end-states — depth 1 = absolute idea, depth 3 = relative `[predicate, idea1, idea2]` | MIXED per row (by depth) | scalar `t−f` in the end-state slot (stage 3); `None` for absolute rows | transient (`persistent=False`, cleared on Reset) | always-on (discourse); trust slot gated `truthIdeas` | inter-sentence AR predictor |

> **"LTM is persisted STM" + "LTM is provisioned, not accumulated"** name the relationship between the
> last two columns and the source of LTM, respectively — they do **not** merge the loci (next note).

### Two consolidation questions (resolved 2026-06-18)

**Q: Can the Truth-Ideas (RelativeTruthStore) "just be added to the LTM"?** The code review found the two
**as built** are two projections for disjoint consumers and recommended keep-separate. **DECISION (Alec,
2026-06-18): CONSOLIDATE — move the RelativeTruthStore corpus INTO the LTM**, and change the LTM (and STM)
so it becomes a suitable single reasoning home. The relative LTM end-state `[predicate, idea1, idea2]` and a
RelativeTruthStore `(np1=idea1, vp=predicate, np2=idea2)` triple are the **same content, co-derived from the
same STM end-state buffer**, so the duplication is real; the review's objections become the **requirements**
the move must satisfy:

- **Not root-reduced.** The LTM already *stores* the full `[depth, D]` end-state and only the AR predictor
  reduces it to a root *on read*. Keep it that way: reasoning addresses the full relation by slot (predicate =
  slot `depth-1`, idea1 = `depth-2`, idea2 = slot 0); root-reduction stays a predictor-local read, never a
  storage policy.
- **Not transient → persistent.** Today `_stm_end_states` is `persistent=False` and cleared on Reset (an AR
  ring). As the reasoning corpus it must **persist** (state_dict / checkpoint) and survive document-boundary
  Resets — universal relations must not age out or reset.
- **Trust on the LTM → and therefore on the STM.** The LTM end-state already carries the scalar-trust slot.
  Alec: *"it will pick up a trust value, but that means STM should also."* So **STM gains a per-idea trust
  channel** — on the **live idea-stack** (`_idea_*` buffers, what `stm.push_step_masked` writes), **not** the
  typed STM (`_category`/`_order`/`_ref_id`, which is driver-only and never written in the forward) — so trust
  propagates STM→LTM. **Caveat (verified):** there is *no per-idea trust source in the forward today* (trust is
  a relation/predicate property computed at the boundary via `_tetralemma_trust`), and the end-state predicate
  slot is a *folded* reduction, so per-idea trust does not map 1:1 to the end-state trust — a source + a
  fold/combine must be defined (see the resolution checklist below).
- **Open items the move must also handle (from the review).** Reasoning must run on the **content-width slice**
  (conform like `_conform_idea_vec`, not the event width — else `.where`/`.when` energy leaks into parthood);
  the reasoning **scope** (a global flat corpus vs per-row recency-evicting deques) must be reconciled; and
  reasoning must **filter to the relation subset** (the LTM records every end-state, absolute and reducible
  included). Once the LTM satisfies these, `RelativeTruthStore` is **subsumed and retired**, and
  `reason` / `consequents` / `evaluate` / `verify_relation` read the LTM. The `_trusts` persistence fix below is
  a step toward "trust persists," consistent with this move.

**LOCKED + FOUNDATION LANDED (2026-06-18, Alec).** The consolidation is a unified **ternary tensor**,
`Layers.TernaryTruthStore` (**stage 1 BUILT** — `test/test_ternary_truth_store.py`, 15 tests; additive /
inert until wired): rows `(NP1, VP, NP2)` of **full idea vectors** (`Null` = zero) + a per-row **timestamp**
+ a scalar **trust**; a `rel_type` tag carries `partOf` / `implies` / other; `NP··` = absolute idea,
`NP VP·` = unary, `NP VP NP` = relation. Stored **UNSCALED** (trust a separate column — no magnitude-baking),
all registered buffers (**persistent** → rides the state_dict). **Scope = LTM + RelativeTruthStore only**
(Alec): TruthLayer stays for luminosity; WS-META stays the *reducible* home (reduce-or-describe survives — its
*ineffable* branch will target this store). Timestamp = monotonic clock (XML rows earliest); new dark gate
`ltmConsolidation`; reasoning scans `relations()`. **STAGES 2–6 DONE (2026-06-18; full suite 2648/0,
flag-off byte-identical; `test/test_ltm_consolidation.py`, 19 tests).** (2) `<ltmConsolidation>` gate
(model.xsd, `Models.ltm_consolidation`, `cs._ltm_consolidation`); `SymbolicSubSpace.ltm_store =
TernaryTruthStore(muxed, content_width=symbol_dim)` built **only** when gated (RTS built only when off — the
class stays for flag-off + the ~19 standalone-store tests); attribute-only (submodule → state_dict; out of
`self.layers` → survives the Reset cascade). (3) the observe site appends one ternary row per batch row from
the same `depths`/`payloads`/`tetralemmas` (depth-1 → `append_idea`; depth-3 → `append_relation` with
NP1=idea1=payload[d-2], VP=predicate=payload[d-1], NP2=idea2=payload[0]), under the existing
`not _exploration_trial` guard; the deque + AR predictor are **left untouched** (the deque is the AR ring; the
store is the persistent record — observe writes both). (4) `_route_learned_relation`'s ineffable branch returns
`('idea', -1)` without a separate write when consolidated (the row already exists from observe); reduce branch
unchanged. (5) `reason`/`verify_relation` select the store (`_reasoning_store`) and read it via
`_iter_relation_rows` (RTS: un-bake `np/t₁`; ternary: unscaled slots + separate trust, content-sliced);
`verify` write-back = `set_trust` (ternary, no re-bake) / re-bake (RTS). (6) RTS retired on consolidated configs;
`provision_ltm()` appends `<truthSet>` rows at load (earliest timestamps). **The three follow-ups are now REAL (2026-06-18; full suite green):**
(i) **Real-parse provisioning** — `provision_ltm` runs each `<truthSet>` text through the actual forward
(`prepInput`+`forward` under `TheData.runtime_batch`; the briefed `runEpoch(split="runtime")` path raises on its
inference fast-path, so the working `_infer_ir`-style path is used), so the append lands a real parsed end-state
(real encoding + real NP/VP/NP for an in-grammar relative parse); the XML `trust` + `kind`→`rel_type` are then
stamped on the appended rows. `_encode_text_to_idea` (mean-pool) and the NP1=VP=NP2 hack are DELETED.
Triggered lazily once at the first `runEpoch` (guard `_ltm_provisioned`; data must be loaded), not at
`from_config`. (ii) **Conversation pushed regardless of a predictor** — the store-append was de-nested out of
the discourse block, so it fires whenever consolidated (the single append path, reused by provisioning).
(iii) **AR reads the store** — `InterSentenceLayer.get_stm_chain` reconstructs end-states from
`ltm_store.recent()` (and `observe` skips the deque when consolidated); the reconstruction is **INFIX**
`[idea1, predicate, idea2]` (the store's native order — idea1 may stand without a predicate) and the consolidated
AR root is **idea1 (slot 0)**, kept consistent between predicted-context and observed end-state.
**Remaining inherent limits (not placeholders):** parse fidelity is grammar/vocab-dependent (out-of-grammar text
yields whatever the parser produces; the `kind` tag still forces `rel_type`); the AR-from-store is global-recency
(B=1 correct; B>1 batched training shares one recency window). **Two pre-existing bugs surfaced (separate
issues):** `Spaces._topk_priming_mask` crashes under `sentenceProtocol`+top-k-priming at small widths
(reproduces on `MM_grammar.xml`), and `runEpoch(split="runtime")` raises on its no-`batch_override` inference
fast-path. The checklist below is the design rationale.

**USER TRUTHSET → LTM (2026-07-10).** The runtime request-body TruthSet (`store_truths`, the serve layer's
`/chat/completions` `truth` entries) is consolidated into the same store: a per-row **`origin`** column
(`ORIGIN_CONVERSATION` / `ORIGIN_PROVISIONED` / `ORIGIN_USER`, a registered buffer) plus host-side source
texts (`set_origin(idx, origin, text=)`, transient like `TruthLayer._sources`) distinguish the three writers.
`store_truths` on a consolidated config compacts prior user rows out (`clear_origin` — replace-on-resubmit;
the client sends its full TruthSet each request), runs each text through the same real-parse ingestion
provisioning uses (`_ltm_ingest_truth_texts`, extracted from `provision_ltm`), and stamps trust + origin +
text on the landed rows — user rows, XML rows and conversation rows coexist in the one persistent store.
The **TruthLayer is no longer the canonical user-truth store on consolidated configs**: it is a
compatibility read model (`attach_ltm` back-ref at construction; `sync_from_ltm` materializes the
provisioned + user rows as one flat content-band vector per row × trust, plus `_sources`/`_trusts`), so
the flat-field readers — luminosity, falsity penalty, consistency, clarifications, assess — read the
LTM-backed data unchanged and the serve surface (clarifications / truth_assessment) is preserved. Gate off:
`store_truths` keeps the legacy epoch-ingestion path byte-identical. Tests:
`test_ternary_truth_store.py::TestOriginProvenance`,
`test_ltm_consolidation.py::{TestTruthLayerLtmView, TestRuntimeUserIngestion}`.

**LOAD-TIME REVIVE, gated by `<stateless>` (2026-07-11).** New architecture flag
`architecture.stateless` (model.xsd, `Models.stateless`, `SymbolSubSpace._stateless`;
**DEFAULT TRUE**, mirroring WikiOracle's `server.stateless` — the shipped deployment runs
stateless). A `register_load_state_dict_post_hook` on `SymbolSubSpace`
(`_revive_ltm_post_load`, registered only on consolidated configs) fires after every
`load_state_dict` (a parent post-hook runs after all children, so `ltm_store` +
`truth_layer` buffers are loaded) and **rematerializes the TruthLayer view** from the
loaded store. Stateless (default): the checkpoint's request-scoped `ORIGIN_USER` rows are
`clear_origin`'d first — each request supplies its own TruthSet, so only the persistent
provisioned + conversation rows revive. Stateful (`<stateless>false`): user rows are durable
state and are kept. Host-side `_texts` don't ride the state_dict, so revived rows read
`text=None` (the source strings are gone after a stateless reload anyway). Tests:
`test_ltm_consolidation.py::TestStatelessRevive` + `MM_ltm_consolidation_stateful_fixture.xml`.

**Resolution checklist before building (2026-06-18, code-grounded — workflow `wf_eb4ccbf5-329`, 6 readers).**
Beyond the three requirements above, these must be settled:

*Decisions (design forks):*
1. **Scope.** Three corpora (TruthLayer stays; WS-META stays the *reducible* home; LTM absorbs only the
   *ineffable* `RelativeTruthStore` branch) — **recommended** — vs one corpus (fold WS-META in too; loses the
   taxonomy/lattice/`part_chain`). Sets the blast radius; decide first.
2. **Partitioned LTM.** One `InterSentenceLayer` with **two partitions** (transient recency ring for AR;
   retained relations partition for reasoning) — **recommended** — vs one undifferentiated deque.
3. **Reasoning scope:** per-row vs **global/pooled** (recommended; matches today's `reason()`).
4. **Filter + dedup:** tag relation rows at write (`rel_mask` is in hand at the observe site,
   [Models.py:7239](../../bin/Models.py)) + a relation-key idempotency index that EMAs trust on re-assert
   (fixes `RelativeTruthStore`'s append-only double-count) — **recommended** — vs append-only.
5. **STM trust SOURCE (murkiest).** No per-idea trust exists in the forward today (trust is a
   relation/predicate property at the boundary); the end-state predicate slot is a *folded* reduction.
   Decide: invent a real per-idea trust (idea activation sign + TruthSet) vs a seed; **feed** vs replace
   `stm_end_state_trust`; define the fold.
6. **Trust shape:** scalar `t−f` (recommended) vs full tetralemma.
7. **Gating:** a **new dark gate** (`ltmConsolidation`, recommended) vs reuse `truthIdeas` — the LTM append is
   already LIVE/ungated on the 4 `sentencePrediction` configs, so persistence/Reset changes alter `L_inter`
   gradients and are NOT byte-identical unless gated.

*Hard code invariants to change:*
- LTM is cleared at **three** sites — doc-boundary `Reset` ([Layers.py:8268](../../bin/Layers.py), currently
  ignores `hard` via `del hard`), epoch-start `ss.discourse.reset()` ([Models.py:4476](../../bin/Models.py)),
  and `ensure_batch` reshape — all must become selective.
- LTM is **not in the state_dict** (ragged per-row deques) → checkpoint **sidecar** (mirror
  `vocab_extras`/`bpe_extras`) + collect/restore; `register_buffer` can't hold it.
- `maxlen` FIFO **eviction** silently drops old relations → the relations partition must be non-evicting.
- **Width:** payloads are event-width `[content|where|when]`; reasoning is content-only → slice content on read.
- **Drop magnitude-baking:** `reason`'s `np/t₁` un-baking simplifies away (LTM stores unscaled + a separate
  scalar trust); `verify_relation` rewrites to overwrite the scalar (deque tuples are immutable → replace the
  entry); `consequents`/`evaluate`/`constraint_residuals` need a flat materialized `[n, content]` view over the
  ragged deques.
- `reason`/`consequents`/`evaluate`/`verify_relation` have **no production callers** (test-only) → re-pointing
  breaks no live path, but retirement repoints **19 tests** (incl. a source-string assert in
  `test_two_truth_stores.py`) and removes the `relative_store.*` state_dict keys (needs a checkpoint migration).

*Recommended suite-green sequence:* (1) non-root-reduced LTM read API (additive) → (2) STM idea-stack trust
channel (gated) → (3) LTM persistence sidecar (gated) → (4) LTM survive-Reset (gated) → (5) reasoning-over-LTM
(port readers; keep `store=` injection) → (6) retire `RelativeTruthStore` last (hard-cut + repoint tests +
checkpoint migration).

**Q: Augment the LTM by parsing a TruthSet from XML config?** **Yes — sound and additive — but DEFERRED
until the consolidation above lands** (provisioning should target the consolidated LTM, not the
soon-to-be-retired RelativeTruthStore). This is the
**provisioning** path the design calls for ("WikiOracle is stateless ⇒ LTM is PROVISIONED, not accumulated;
the live pipeline is identical, only the source of LTM differs"), and the small-scale form of the deferred
"pre-parse Wikipedia" TODO. **Current state (verified):** there is **no** XML/data-file truth loader today —
user truths arrive as a **runtime JSON payload** (`serve.py` body `'truth'` → `store_truths`); XML supplies only
the *knobs* (`truthCriterion`, `truthMaxEntries`, the luminosity/universality weights). So an XML `<truthSet>`
(or a config-referenced truth file) parsed at model load and fed through the **same** serial parse →
reduce-or-describe routing → LTM-observe pipeline (so each truth lands in its proper loci above:
reducible→WS META, ineffable→RelativeTruthStore, end-state→LTM with trust) is a genuinely new, additive
capability — recommended as the concrete next build.

> **Bug surfaced by the review — FIXED 2026-06-18.** `RelativeTruthStore` kept per-triple trust in a plain
> Python list (`_trusts`) **outside** the state_dict, while the trust-baked magnitudes (`np1/vp/np2`) were
> registered buffers — so on checkpoint reload the per-relation trust silently reset to the `1.0` fallback in
> `reason`/`verify_relation`. **Fixed** by making trust a registered `[max_triples]` buffer
> (`RelativeTruthStore.trust`, [Layers.py:7348](../../bin/Layers.py)): `record_triple` writes it, `reset` zeros
> it, and `reason`/`verify_relation` read/write it
> ([Spaces.py:13692](../../bin/Spaces.py) / [13761](../../bin/Spaces.py)); a read-only `_trusts` property is
> kept for back-compat. Old checkpoints load non-strict (buffer keeps its zero-init). Tests:
> `TestRelativeTrustPersistence` (4). (The absolute `TruthLayer` has the same list-not-in-state_dict pattern
> for `_sources`/`_trusts`, but that is clarification-surfacing metadata with separate semantics — left as-is.)

### Correctness partition — ENTITIES are absolute, LINKS are relative
- **Absolute (non-relative):** *codes and ideas, at ANY order.* Zero-order codes/ideas are absolute;
  higher-order ones are **abstract but still absolute** (non-relative — loosely "no symbol is absolute,"
  but these are non-relative). A higher-order code's constituents (its synthesis/definition) ride WITH
  the code (absolute). Codebooks stay **assumed-valid** (no per-proposition trust).
- **Relative (trust-bearing):** *relations BETWEEN codes or ideas, at any order* — the asserted links
  (A→B, cat ⊑ animal). Testimony writes **relations**, never entities; it is never promoted into the
  assumed-valid codebook (that would treat testimony as direct experience).

### Lifecycle combine rules (the over-collection remedies)
- **Too many parts → combine with AND** (σ-synthesis: a higher-order part is the AND/intersection of
  its particles — narrows). Live: `ConceptualSpace.synthesize_higher_order` / `refine_over_collected`.
- **Too many wholes → combine with OR** (π: a broader whole is the OR/union of properties — widens).
  The dual combine — NOT a split. **Deferred ("nothing to do now; going forward we need more wholes").**
  (`PiLayer.factorize_over_set` is built as the π inverse primitive, available for later.)

### Reducibility (route a parsed predication)
Try to reduce the parse to symbols (run-it-and-see via the σ/π compose). If it reduces → a relation
over **codes**. If it is **ineffable** (won't reduce) → approximate with a **longer description** (a
more-composed **idea**) → a relation over **ideas**. (Composed NPs like "bowling pins" are ideas, not
single codes.)

### Idea identity
Ideas are compared by **shared parts and wholes** — identical *to the degree* they share them (graded,
collection-based, like symbol retrieval). No exact-match requirement.

### Reasoning (forward inference; expands the area of luminosity)
Modus-ponens over relations. Given a relation **A→B** (trust `t₁`) and a known truth **C** (trust `t₂`):
1. **Map C to the antecedent A by parthood** — direct match, OR raise/lower C or A to a common order to
   test whether **C is covered by A**;
2. if C satisfies A, **apply the consequent VP (B) to C → a NEW conceptual space (a new concept)**;
3. the new concept's trust = **`t₁ × t₂`** (illuminated to the degree both are true);
4. the **area of luminosity** (the illuminated / known-true region) grows.
(Reasoning is the tier after authority; experience/episodic verification feeds trust separately.)

### STM → LTM
LTM **is persisted STM**:
- **Serial mode:** an **absolute** truth = **1 position** (one idea/code); a **relative** truth =
  **2 positions** (the two related entities — *parthood over extension*; 2nd position often empty for
  absolute truths).
- **Parallel mode:** the full **N (≈8)** STM ideas.
- **Over time:** LTM = the **stack of STM** (keeps both verbal/serial and nonverbal/parallel memory);
  minimally just positions 1 & 2.
- WikiOracle is **stateless ⇒ LTM is PROVISIONED** (pre-parsed Wikipedia → ideas/relations + per-item
  trust), not accumulated. The live pipeline is identical; only the source of LTM differs.

### Pipeline
serial parse → **reduce-or-describe** (codes if effable, longer-description idea if not) → tag **trust**
→ store the **relation** (relative) over the **entities** (absolute) as STM positions → **persist
STM→LTM** → **reasoning** (modus ponens + parthood-match + `t₁×t₂` + luminosity) → **verify** (episodic
order-0 instances → trust).

### Handoff build stages
1. **✓ DONE (2026-06-18). Map** the existing `RelativeTruthStore` / `TruthLayer` /
   `_maybe_learn_relation` / `learn_relations_from_stm` APIs. **Key finding: the two-store design
   already EXISTS and is unwired** — `test/test_two_truth_stores.py` documents it (absolute=IDEAS in
   `TruthLayer`; relative=RELATIONS-BETWEEN-IDEAS as uncollapsed np1/vp/np2 in `RelativeTruthStore`,
   the latter with ZERO production callers). `RelativeTruthStore.consequents()` (Layers.py:7378) IS
   the modus-ponens forward step; `learn_relations_from_stm` (Spaces.py:13848) is LIVE at
   `ConceptualSpace.Reset(hard=True)` reading the depth-3 STM end-state; `InterSentenceLayer.
   observe_stm_end_state` (Layers.py:7870) already persists STM end-states with a `tetralemmas` slot
   parked at `None` (the trust hook). No `.events` episodic store exists. Three decisions (Alec):
   **two homes by reducibility** ("the intuitive and explicit knowings"); **trust = full tetralemma
   during computation, stored as a scalar `t − f` ∈ [−1,1]** (Dharmakīrti — BOTH/NEITHER are not
   objects of knowing); **reducibility = codebook-snap** (both entity operands snap to existing rows,
   no mint).
2. **✓ DONE (2026-06-18; suite 2594/0, flag-off byte-identical). Store routing + trust** — gated
   `<truthIdeas>` (parsed `Models.truth_ideas`; stamped `cs._truth_ideas`). `_maybe_learn_relation`
   now routes via `_route_learned_relation`: REDUCIBLE (`_relation_is_reducible` = both operands snap)
   → WS META `insert_relation` with the FULL tetralemma (the *intuitive* knowing); INEFFABLE → the
   uncollapsed `(idea1, predicate, idea2)` triple into `RelativeTruthStore` with the scalar trust
   `_collapse_trust` (`t − f`) (the *explicit* knowing; returns `('idea', row)`). `_conform_idea_vec`
   reconciles the idea (content) width vs the store (event) width. Tests
   `test/test_truth_ideas_routing.py` (9).
3. **✓ DONE (2026-06-18). STM→LTM persistence** — the learned per-row SCALAR trust is attached to the
   LTM end-state's `tetralemmas` slot. `ConceptualSpace.stm_end_state_trust(buf, relative_mask)`
   computes, per relative row, `_collapse_trust(_tetralemma_trust(predicate))` (the relation at slot
   `depth-1`); absolute rows → `None`. **Ordering finding:** the live hooks run `observe` (in the
   forward, [Models.py:7269](../../bin/Models.py)) BEFORE `learn` (in the post-batch `dispatch_per_row_
   reset` → `CS.Reset`), so a learn-time stash would lag by a boundary; the trust is therefore computed
   on the CS from the SAME end-state buffer `observe` records and read there (`tetralemmas=` arg),
   stashed on `cs._last_end_state_trust`. Gated `truthIdeas`; `None` when off (slot stays `None` →
   byte-identical). The 1/2-position serial vs N-stack parallel persistence is the existing end-state
   depth (1 absolute / 3 relative). The **provisioning format** (pre-parsed Wikipedia) remains a future
   TODO (Workstream G).
4. **✓ DONE (2026-06-18). Reasoning engine** — `ConceptualSpace.reason(query, query_trust, *,
   parthood_threshold, max_steps, store)`: forward modus ponens over the relative store. For each
   relation whose antecedent `A` covers the query `C` by parthood (`_idea_parthood` ≥ threshold; `A`/`B`
   recovered UNSCALED since `record_triple` bakes trust into the magnitude), apply consequent `B` → a new
   concept with trust `t₁×t₂`; returns `{'derived': [...], 'luminosity_gain'}` (the illuminated area).
   `max_steps>1` forward-chains (each relation fires once). `_idea_identity` is the graded
   shared-parts/wholes idea match. Order raise/lower reduces to the graded parthood coverage on the idea
   vectors (a first cut; explicit order normalization is a documented extension).
5. **✓ DONE (mechanism; 2026-06-18). Verification** — `ConceptualSpace.verify_relation(relation_idx,
   episodes, *, store, parthood_threshold, support_weight)`: order-0 episodes (antecedent/consequent
   idea pairs) that cover `A`/`B` by parthood nudge the stored scalar trust toward observed support
   (`new = (1-w)·old + w·(2·support_frac − 1)`, clamped; magnitude re-baked). **The episodic SOURCE**
   (persisted `.events` exemplar store / provisioned Wikipedia LTM) is the remaining FUTURE TODO
   (Workstream G, no owner); this is the verification mechanism that consumes whatever episodes are
   supplied.
6. **(deferred nice-to-haves, unchanged)** the too-many-wholes OR-combine; the π-split (its source
   `PiLayer.factorize_over_set` is built); the σ tower-codebook geometric realization
   (`SigmaLayer.synthesize_over_set`). Per Alec ("unneeded complexity for the moment").

All of stages 2–5 are behind `<truthIdeas>` (the reasoning/verify query methods are inert until the
store is populated, which only the gated learn path does); flag-off is byte-identical (full suite). Tests:
`test/test_truth_ideas_routing.py` (25).
