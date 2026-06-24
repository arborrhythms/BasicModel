# The Conceptual Space — a Gärdenfors similarity layer between mereonomy and taxonomy

> **Design.** The architecture has two ways of relating concepts — *mereonymic*
> (part/whole composition) and *taxonomic* (is-a subsumption) — but **no
> well-defined similarity space**. In Gärdenfors' terms the missing layer is a
> **conceptual space**: a geometric *similarity* space, neither mereonymic nor
> taxonomic, where substitutable concepts are co-located. This doc captures the
> three-layer model, why the similarity layer is the significant absence, and the
> mechanism that fills it: **SBOW-situating the concept *codes*** — which is the
> move from a symbolic **table of indices** to a **vector space**.

## 1. The thesis: the absent middle layer

A concept can be related to others in three different *kinds* of way. The system
represents two of them and is missing the third:

| Layer | Relation | Character | Geometry? | Status |
|---|---|---|---|---|
| **Mereonymic** | part-of / composition ("the black cat" = whole over {the, black, cat}; a word over its bytes) | bound by **spatiotemporal locality** (`.where`/`.when`) — the binding problem | geometric, **non-syntactic** | present (PartSpace ⊕ WholeSpace) |
| **Conceptual (similarity)** | *how alike* / *what substitutes for what* (cat ~ dog ~ animal) | **distributional locality** — siblings define position | geometric, **syntactic** (substitutability) | **ABSENT** |
| **Taxonomic** | is-a / property inheritance (cat ⊑ mammal; "has-fur" at the mammal level) | **set-based** subsumption | **non-geometric** (symbolic sets) | present (relations / property-at-level) |

The mereonymic layer answers *what is X made of / where is X*. The taxonomic
layer answers *what kind of thing is X* (and, for concepts held **only by
symbol**, it is the *only* relational knowing available). Neither answers *what
is X **like** — what could stand in for it*. That is the **conceptual /
similarity** layer, and it is what word2vec captures for words. It is the
significant absence.

## 2. The conceptual space is a Gärdenfors similarity space

Gärdenfors' *conceptual spaces* are exactly the right type for this layer, and he
places it **between** the symbolic and the subsymbolic — which is precisely our
three-layer split:

- **Quality dimensions** + a metric: **similarity = distance**.
- **Concepts are convex regions** (Criterion P / betweenness: if x and y are in a
  concept, so is anything between them).
- **Prototypes are the region centers**; typicality falls off with distance.
- Substitutable items are **co-located**; relations show up as **offsets**
  (the analogy structure of an empirical conceptual space).

word2vec is, in effect, an *empirical* Gärdenfors space; **CBOW/SBOW is how you
instantiate one** (a concept is placed by the bag of its context — its siblings).

**One deliberate departure from textbook Gärdenfors** (and the right one, per the
cognition): he geometrizes is-a as **region containment** (sub-concept ⊆
super-concept). We keep **subsumption set-based** instead. So the conceptual
space carries *similarity / substitutability / typicality* (geometric), while the
**taxonomy carries inheritance** (symbolic, sets). **"Gärdenfors for *how alike*,
sets for *what kind of*."** This preserves the asymmetric, non-metric character
of is-a — and the cognitive evidence that human categories behave like feature
sets, not points in a clean metric — which a distance cannot honor.

## 3. The significant move: indices → codes (a table becomes a vector space)

This is *not* a small build. Today **concept identity is the row index**: "codebook
identity is the row index (the `_index` selection)" and "the taxonomy is
row/position-keyed" (`bin/Spaces.py:927, 936, 1001, 2774, 3012`). Concepts are
**pointers into a table**. A table has *no* similarity, *no* prototype, *no*
betweenness — only equality of indices and whatever explicit pairwise edges are
stored beside it.

The refactor replaces the index with a **code** — a vector with a definite
**location** in the conceptual space. Once a concept is a located vector:

- *similarity* is a distance (cheap, differentiable),
- *prototype* is a region center,
- *substitutability* is proximity,
- *betweenness / interpolation / analogy* are vector operations,

— none of which a table can express. **We are moving from a symbolic lookup table
to a similarity space.** That is the essence of the refactor; the SBOW call is
the *mechanism*, but the representational shift is the *substance*.

## 4. The mechanism: SBOW-situate the concept-codes

During the **parallel pass at word level** (`symbolicOrder=0`, all words present
at once), run an **SBOW (sum/bag-of-words) encoding over the concept-codes** —
i.e. each concept-code is positioned by the bag of its **siblings** (the other
concept-codes co-present in the window). This is CBOW: a concept's location is a
function of its context, so concepts sharing siblings (sharing context) land
together → **substitutability geometry**.

- The primitive already exists at the perceptual level: `sbow_loss`
  (`bin/Spaces.py:4920`; consumed in `runBatch` via
  `self.perceptualSpace.sbow_loss(words)` / `perceptual_sbow_loss`,
  `bin/Models.py:3250, 3257, 4175`). The refactor **points it at the
  concept-codes** (the conceptual `.what`, `bin/Spaces.py:5594), not only the
  percepts.
- **Words and objects both get situated.** For words, siblings are the other
  words in the window. For objects known *first-hand* (perceptually grounded),
  siblings are the other objects co-present spatiotemporally — the same SBOW
  situating, so words and objects are both well-placed in one space.
- **Dense vs discrete.** Similarity wants a smooth continuous space; the codebook
  is VQ. Situate the **pre-snap continuous** code and keep that codebook
  low-frequency / EMA, so a concept carries a *smooth, mineable* location **and**
  snaps to a symbol — the snap feeds the set-based taxonomy, the vector feeds the
  geometry.

## 5. The contracts between the three layers

The layers are different kinds of structure, composed by three distinct
operations — and the boundaries between them are load-bearing (the reasoner
already forbids composing taxonomic subsumption with mereonymic transitivity):

1. **Bind** (mereonymic): parts + properties → a concept, via **spatiotemporal
   locality** (`.where`/`.when`). *Solves the binding problem.*
2. **Situate** (conceptual): a concept → a **location** in the similarity space,
   via **distributional locality** (SBOW of siblings). *Gives substitutability,
   prototypes, similarity.*
3. **Subsume** (taxonomic): a concept → its **type**, via **set-based**
   property-at-level inheritance. *Gives is-a for symbol-only concepts.*

Binding builds the concept; situating places it; subsumption classifies it. Each
is needed; none reduces to another; and the conceptual space is the one that has
been missing.

## 6. What exists vs. what changes

| Piece | Where | Change |
|---|---|---|
| Concept identity = row index (the "table") | `bin/Spaces.py:927, 936, 1001, 2774, 3012` | **→ a located code** (the core shift) |
| `sbow_loss` (perceptual) | `bin/Spaces.py:4920`; `Models.py:3250, 3257` | **point it at the concept-codes** in the parallel pass |
| Conceptual `.what` codebook | `bin/Spaces.py:5594` | becomes the **similarity-space home**; trained pre-snap by SBOW |
| Mereonymic composition (σ/π, `.where`/`.when`) | PartSpace ⊕ WholeSpace | **unchanged** — still the binding layer |
| Taxonomic relations / property-at-level | the truth/relation store | **unchanged** — stays set-based |

## 7. Design choices / open questions

- **Code width (the bottleneck).** The conceptual `.what` dimension is the
  factorizing bottleneck — wide enough for the quality dimensions, narrow enough
  to force generalization. (A narrow continuous `.what` under the SBOW objective
  is what makes the geometry mineable rather than a raw co-occurrence vector.)
- **Window / "sibling" definition.** Whole parallel-pass window vs a bounded
  neighborhood; for objects, the spatiotemporal co-presence set.
- **Migration (index → code).** Existing checkpoints key concepts by index;
  introducing a located code is a representation change (regenerate or migrate;
  the index can remain as the *symbol* the code snaps to).
- **Convexity pressure.** Whether to add an explicit Criterion-P/convexity
  regularizer, or let SBOW + the EMA codebook produce convex regions implicitly.
- **Boundary between layers.** Keep the conceptual-space proximity (similarity)
  strictly separate from the taxonomic subsumption (sets) so the part-whole /
  is-a transitivity fallacy stays impossible by construction.

## 8. Dimensionality: three currencies (perceptDim / conceptDim / symbolDim)

Once a concept is its own situated **code** (§3) rather than an index into the
percept representation, its dimensionality stops being forced to match the
percept's. The three representational *currencies* become independent:

- **perceptDim** — the mereonomic content (`.what`) of PS ⊕ WS (part/whole form).
- **conceptDim** — the ConceptualSpace codebook (the similarity space).
- **symbolDim** — the SymbolSpace compressed symbol/index.

**Thin mereonomy, fat conceptual space (for text).** Form and meaning have
*opposite* complexity in text: the orthographic mereonomy is low-complexity (a
~256-symbol alphabet, short sequences, a bounded morpheme set) and wants a
**small** perceptDim; the conceptual similarity space must hold 10⁴–10⁶ word
meanings with usable similarity geometry and wants a **large** conceptDim. So
`perceptDim ≪ conceptDim` is the right allocation for text, and the
percept→concept step becomes an **embedding/expansion** (form-identity → rich
meaning). This is **text-specific**: for grounded objects/vision the mereonomy
(spatial extent, visual parts, the `.where` manifold) is itself high-dim —
per-space dims are exactly what let text run thin while a perceptual modality
runs fat. **Floor:** keep perceptDim high enough that the σ-fold keeps the bounded
part-prototypes distinct (G1 recoverability) — modest for text (thousands of
morphemes fit in ~64–128 dims), but not arbitrarily small.

**Grabbed, not chained.** These three already exist *internally* —
`percept_dim` / `concept_dim` / `symbol_dim` are passed as constructor args
(`SymbolSpace` `bin/Language.py:9077`; the STM `:8849`; `concept_dim` `:9532`; the
symbol codebook `:9188`) — but are **derived** from the chained per-space dim
fields (`_resolve_dim` walks `percept_event → concept_event → symbol_event`,
`bin/Models.py:5456–5464`), with each width declared **twice** (one space's
`nOutputDim` = the next's `nInputDim`) and kept consistent by hand. The refactor
**promotes them to first-class config knobs** — `<perceptDim>` / `<conceptDim>` /
`<symbolDim>` defined once, **grabbed** by PS/WS/CS/SS per their role:
- **PS, WS** → perceptDim;
- **CS** → perceptDim on I/O, conceptDim in its codebook (*the split that isn't
  cleanly passed today*);
- **SS** → conceptDim in, symbolDim for the compressed index.

This removes the duplicated `nInputDim`/`nOutputDim` plumbing and the
chaining-consistency burden, and makes the two dim **transitions** explicit:
perceptDim→conceptDim is the embedding (PS/WS → CS), conceptDim→symbolDim is the
compression (CS → SS) — exactly where the adapters live.

**Scope.** These are the `.what` **content** dims, not the event width
(`.what ⊕ .where ⊕ .when`; the where/when channels keep their own small fixed
widths via the `Encoding` classes, `bin/Spaces.py:911, 1050, 1397`). The
**counts** (`nVectors`, the position counts `nInput`/`nOutput`) are a separate
axis and stay per-space. The pipeline **topology** (PS/WS ↔ CS ↔ SS) must be
named once, since it is no longer implied by matched dim fields. Migration is
aliasing: take perceptDim/conceptDim/symbolDim as the source of truth, derive the
legacy `nInputDim`/`nOutputDim` from them + the topology, deprecate the per-space
dim fields. (They may also already be partly surfaced in the `Encoding` classes /
`ModelFactory` — fold those in rather than re-deriving.)

## 9. Relationship to the other workstreams

- **Mereonomy / decode (Track 1).** The σ-fold recoverability (the
  `2026-06-23-decode-roundtrip-track1.md` G1 work) is the binding layer's
  invertibility; unchanged here.
- **Taxonomy / reasoning.** The truth-grounded reasoner
  (`2026-06-23-reasoning-live-wiring.md`) operates on the set-based taxonomy; the
  conceptual space is a *soft, differentiable* complement that **proposes/orders**
  by similarity, with the hard deduction still standing on the symbolic edges.
- **The predictor.** The contrastive next-idea loss (the trial-split predictor)
  and SBOW are the same distributional family — situating concepts vs predicting
  the next idea — and can share the bottleneck/objective once the concept-codes
  exist.

## 10. Why it matters

Words' significance for word2vec is **not** their part-whole structure; it is
their position in **syntactic / similarity space** (what substitutes for what).
Objects, known first-hand, deserve the same grounding. The system has the
mereonymic and taxonomic layers but not the similarity layer, so it cannot
represent *likeness* or *substitutability* at all — the richest, most mineable
thing about a learned semantics. Introducing a **well-defined Gärdenfors
conceptual space, populated by SBOW-situating concept-codes**, supplies exactly
that missing middle — and does it by turning a **table of indices** into a
**vector space**.
