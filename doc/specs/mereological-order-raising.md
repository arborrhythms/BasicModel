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
  and whole**, and is associated with **both**: `insert_meta(ps_pos, ss_pos)` binds a
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

### How analysis/whole types integrate with the `(ps,ss)` meronomy (2026-06-16, Alec — RESOLVED)

The integration question — "how do WholeSpace analysis/property wholes (which have `.where`
regions but are not PS percepts) connect to the `(ps,ss)` meronomy?" — is answered by the
type/instance/`.where` distinction:

- A WholeSpace **word** whole is ONE `.what` code (the word TYPE); the several word-instances in a
  sentence are that one type at different `.where`s — exactly as the PartSpace **`A`** is one
  `.what` code (the letter TYPE) with many `.where` instances. Neither is "actual" (a glyph in a
  font). The word-type may even **reconstruct better** than a letter-part (its constituent letters
  average to the type mean M) — the code is the stable thing.
- **The meronomy edge is TYPE → TYPE:** "is `A` a part of *word*?" means "do we store the PS
  `A`-code as a part of the WS word-code?" — a single edge between the two TYPE codes in the
  `(ps,ss)` taxonomy.
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

**Critical path:** B1 → B2 → A4 → C → A5 → D (the perceptual mereology end-to-end). F + G (the
symbolic/relational + memory tracks) are largely independent and proceed in parallel. The next
concrete code step is **B1** (char-class property-tiling).

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
