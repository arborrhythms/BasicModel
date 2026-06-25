# The mereological algorithm

> **The main algorithm.** A single **synthesizer** (parts → wholes) and a single
> **analyzer** (wholes → parts), each monotone, together build a codebook whose
> part/whole codes carry a **partial order by construction** — a *concept
> lattice*. This is the canonical pair; the older `synthesis` modes
> (`radix`/`lexicon`/`bpe`/…) and the `<lexer>`/`<analysis>` cuts are retired
> into it. Selected by `<synthesis>mereological</synthesis>` and
> `<analyzer>mereological</analyzer>`.

```
<PartSpace><synthesis>mereological</synthesis></PartSpace>
<WholeSpace><analyzer>mereological</analyzer></WholeSpace>     <!-- analyzer replaces lexer -->
```

This doc states the algorithm. Companion docs: percept presence geometry
[percept-hypercube.md](percept-hypercube.md); the part/whole laws
[Mereology.md](Mereology.md); the free conceptual layer on top
[plans/2026-06-23-conceptual-similarity-space.md](plans/2026-06-23-conceptual-similarity-space.md).

---

## 1. The two operators (and why they're one design)

Parts and wholes form a lattice. The two operators are its adjoints, built from
opposite poles — the `ATOM` (⊥ = all-`0`, nothing) and `UNIVERSE` (⊤ = all-`1`,
everything) sentinels the codebase already names ([bin/Layers.py](../bin/Layers.py)):

| | **synthesizer** | **analyzer** |
|---|---|---|
| direction | parts → whole | whole → parts |
| pole / start | **⊥ = all-0** (nothing) | **⊤ = all-1** (everything) |
| operation | **join** `0 ∨ p₁ ∨ p₂ ∨ …` | **meet** `1 ∧ c₁ ∧ c₂ ∧ …` |
| produces | **dominating** wholes (suprema) | **subordinate** parts (infima) |
| monotonicity | increasing — `whole ⊐ parts` | decreasing — `part ⊏ whole` |
| config | `<synthesis>mereological</synthesis>` | `<analyzer>mereological</analyzer>` |

They are **exact De Morgan duals under the `1−x` complement**:
`analyzer = ¬ ∘ synthesizer ∘ ¬`. So there is **one** algorithm — the
join-from-⊥ synthesizer — and the analyzer is its complement-mirror. Implement
one, reflect for the other.

These are the σ/π *lattice roles* (union-from-0 / intersection-from-1). They are
**distinct** from the existing `SigmaLayer`/`PiLayer` butterfly folds, which stay
downstream and produce **abstractions** (higher orders). The mereological
synthesizer/analyzer build the *order itself*, on the codes.

---

## 2. The order lives in the codes (no SBOW here)

The deliverable is that the **part/whole codes themselves are monotone on
parts/wholes**: a whole's code dominates its parts' codes, per coordinate, on the
`[0,1]` presence hypercube. That ordered codebook **is** the mereological
embedding — a concept lattice (in Formal Concept Analysis terms, the monotone
operators are the Galois closure; the codes *being* monotone is what makes them a
lattice rather than an unordered table).

This is non-negotiable because **SBOW is excluded from mereological space**. SBOW
(distributional situating) orders the *conceptual* layer of free relations; the
monotone codes order the *mereological* layer of parts/wholes. Drop both and the
part/whole codes have no order at all. SBOW would also *fight* the closure, so it
is not used here and could not be.

---

## 3. The part spec is `.where`; the order is byte-span containment

Parthood is **byte-span containment**: `P ⊑ W ⟺ span(P) ⊆ span(W)`, read directly
off the `.where` brackets (`record_cross_tower_meronomy`, `RunStructureLayer`
`contained_mask`). This is exact and **complete for contiguous parts** — crucially
including **cross-boundary** parts the local build never sees:

```
"abcd" = [0,4]   "ab" = [0,2]   "cd" = [2,4]   "bc" = [1,3] ⊆ [0,4]  ⟹  bc ⊑ abcd
```

So the synthesizer that builds `abcd` from `ab ∨ cd` is *locally blind* to `bc`,
but the **global `.where` poset knows `bc ⊑ abcd`.** That is the full part spec —
nothing is missing — and it is what the order projection (§5) needs.

(Non-contiguous parts like `"ac"` are not `.where`-contiguous and sit outside this
poset — a deliberate boundary, since mereology here is contiguous spans.)

---

## 4. Synthesizer: build, then project

```
synthesize(parts):
    seed  = ⊥(all-0) ∨ part₁ ∨ … ∨ partₖ        # join of the BUILD parts (dominates them)
    poset = { (W,P) : span(P) ⊆ span(W) }        # over promoted sub-parts, from .where
    code  = project_monotone(seed, poset)        # redistribute to dominate ALL parts
```

The **join** of the build parts gives a whole that dominates *those* parts by
construction, but may violate cross-boundary ones (`bc`). The **projection** (§5)
repairs it.

`whole = 0 ∨ Σ parts` is a soft-OR on `[0,1]` presence; on a **sparse grounding**
(few coordinates active per base code) at **bounded depth** (characters → word,
not unbounded), it stays in the interior — off the `[1,1,…]` corner. The
saturation worry is the *cross-order* monotone stack, which is **not** done here:
abstraction (cross-order) is the separate butterfly σ/π, not this synthesizer.

---

## 5. Redistribution = isotonic projection onto the `.where` order

"Redistribute codes when they violate the meronymic constraints" has a precise,
convergent form. The constraint `code(W)_i ≥ code(P)_i` for every containment edge
is **isotonic** and **decouples per coordinate**, so:

> **redistribution = per-coordinate isotonic regression on the `.where`
> containment DAG** — project the codes to the *nearest* point with
> `whole ⊐ part` on every edge.

It is convex and minimum-norm (moves only what it must — no gratuitous
corner-drift), and it has every edge it needs because `.where` *is* the part spec
(§3). Verified end-to-end: a whole seeded as `join(ab, cd)` violating `bc`
(2/8 coords, 23/25 edges) projects to **0 violations — `abcd ⊐ bc` and all edges —
bounded movement, still in `[0,1]`.** A newly promoted part only needs its
`.where`-neighborhood re-projected, not the whole poset.

The **soft** version is a Vendrov-style **hinge** (`Σ max(0, partᵢ − wholeᵢ)`)
drawing the same `.where` edges; with training in the loop the clean form is
**projected gradient** — train the content for the situational embedding, project
onto the order each step (hinge = pressure, projection = guarantee).

(Box embeddings would give containment by construction, but the cross-boundary box
still has to be grown using the same `.where` spec, so they reduce to this;
isotonic projection is the leaner realization.)

---

## 6. Analyzer: the `1−x` mirror, and it replaces the lexer

The analyzer is the synthesizer reflected through `1−x`: **meet-from-⊤** producing
subordinate parts. Concretely it cuts a whole into parts by boundary properties —
the first approximation is the **word splitter = OR(space, punct)** (already live
as the `word`/`analyse` cut; `char_class_region([WHITESPACE, PUNCT])`), which fixes
the word level by fiat. The fuller analyzer grows by AND/OR of a **relational
left/right-of-X** property basis (composable, factorable, monotone), and a
newly-promoted whole is split until it reaches the word level.

`<analyzer>` **replaces `<lexer>`**: tokenization is analytic cutting, owned by the
analysis side. `<analyzer>mereological</analyzer>` selects this algorithm;
`<lexer>` is the deprecated spelling.

---

## 7. The convergence drive (what the word level *is*)

With `subsymbolicOrder > 0`, the two operators iterate: the **synthesizer chunks
parts up**, the **analyzer splits wholes down**, each pass re-feeding "parts still
too small" and "wholes still too big," and they **stop where they meet — the word
level.** This makes the word level *operational* (the fixpoint of the two
operators), not a fiat. The general question — the *right* level (Rosch's basic
level) — is left to learning; the `OR(space, punct)` splitter pins the word level
as a first approximation.

---

## 8. What it retires

One synthesizer, one analyzer — the single Galois adjoint pair. Retired into it:
the alternate `synthesis` modes (`radix`/`lexicon`/`bpe`/`mphf`) and the separate
`<lexer>`/`<analysis>` cuts. SBOW stays strictly on the **conceptual** layer above
(free relations), never on the mereological codes.

---

## 9. Status

Building. **Item 0** (the load-bearing piece): the synthesizer produces whole codes
that dominate their parts — `join-from-⊥ seed + isotonic `.where`-projection` —
fully specified and demonstrated (§5). The analyzer's relational basis, AND, and
factor (§6) and the convergence drive (§7) follow. The existing butterfly `σ/π`
(abstractions) and the conceptual SBOW are unaffected.

---

## References
- ⊥/⊤ poles (`ATOM`/`UNIVERSE`), char-class regions: [bin/Layers.py](../bin/Layers.py)
- `.where` containment, `record_cross_tower_meronomy`, `stage_analysis_spans`: [bin/Spaces.py](../bin/Spaces.py)
- presence geometry: [percept-hypercube.md](percept-hypercube.md)
- part/whole laws, σ/π: [Mereology.md](Mereology.md)
- conceptual layer / SBOW: [plans/2026-06-23-conceptual-similarity-space.md](plans/2026-06-23-conceptual-similarity-space.md)
