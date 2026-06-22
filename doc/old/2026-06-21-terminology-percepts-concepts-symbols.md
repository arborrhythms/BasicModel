# Terminology simplification — percepts / concepts / symbols

**Status:** PLAN (2026-06-21). One noun per tier. This is the **plan-pass**; a
**doc pass** (rename in the live docs) follows soon, and a careful **code-identifier
pass** later (after the SymbolSpace structural refactor lands, to avoid churn-on-churn).

## The convention

| Tier | Space(s) | Thing = | Character |
|---|---|---|---|
| Perceptual | `PartSpace`, `WholeSpace` (both `PerceptualSpace`) | **percept** | dimensionally-embedded (TALL, `.what/.where/.when`), EXTENSIONAL. part-percept = atom (bottom-up σ); whole-percept = property/region (top-down π) |
| Conceptual | `ConceptualSpace` | **concept** | a relation tying ONE part-percept ↔ ONE whole-percept (by reference); the Concept codebook; DIACHRONIC |
| Symbolic | `SymbolSpace` (`PerceptualSpace`) | **symbol** | 0-D (1-D collected), NOT dimensionally embedded; REFERENCES a concept; the Symbol codebook; INTENSIONAL |

Rule of thumb: **percept = observed form, concept = a part↔whole relation, symbol =
a 0-D reference to a concept.** "symbol" stops being a catch-all.

### Meta-concepts — Word, Object, and the taxonomy (Alec 2026-06-21)

ONE uniform structure across orders: **a concept is a `[part, whole]` pair.** At
order 0 the slots hold **percepts**: `Concept = [part-percept, whole-percept]`.
Higher orders reuse the SAME shape with concepts in the slots:

- **Word** and **Object** are each a **concept** (`[part, whole]` of percepts; Object
  spans the ATOM↔UNIVERSE poles), each represented by a **single symbol**.
- A **meta-concept** keeps the structure: `Meta-concept = [Object, Word]` =
  `[part = Object, whole = Word]`. The Word (label/category) is the **whole**, the
  Object (referent/instance) is the **part**, so `Word ⊇ Object` is the taxonomic
  containment. Its **meta-symbol** is the single symbol for that meta-concept (the
  Word≡Object identification; the existing `create_word_object_meta`).
- **Recursion up to `symbolicOrder`:** symbols are **stored as parts and wholes** of
  higher-order `[part, whole]` concepts — collecting symbols this way forms
  **mathematical sets** whose nesting is a **taxonomy** (symbolic subsumption,
  distinct from the order-0 `.where` percept meronomy).

Spine (uniform `[part, whole]` at every order): **percept →
`Concept=[part,whole]` (Word, Object) → one symbol each → `Meta-concept=[Object,Word]`
→ symbols-as-parts/wholes → sets → taxonomy**, `symbolicOrder` bounding the
recursion. This matches the CS Concept codebook's `[_sym_parts, _sym_wholes]` (one
part-ref + one whole-ref per concept) — uniform across orders.

**1:1 concept↔symbol via higher-order collapse.** A concept is a *single* part-whole
connection, so a thing makes a **many:one** asymmetry (many parts under one whole, or
many wholes under one part — the "many" co-present at the same `.where` + `.when`). The
collapse creates a **higher-order mereological part or whole (a SUPERSET)** that
**subsumes** the members (they stop appearing independently), replacing the many:one
symbolic relationship with **one:one → 1:1 concept↔symbol, no lookup**. This *is* the
mereological order-raising; the same collapse builds the Object, the meta-concept
`[Object, Word]`, and the taxonomy.

## The overloading to remove

Today "symbol" means three different things; disambiguate:
1. **CS `_sym_parts`/`_sym_wholes`/`reify_relation`/`new_symbol`/`symbol_is_identity`
   ("the symbol table")** — these are **CONCEPTS**, not symbols. *Biggest misnomer.*
2. **WholeSpace `symbol_dim` / the compact `nOutputDim=8` emission** — the emission
   is becoming the **SYMBOL** (migrating to SS in the refactor); WholeSpace itself
   holds **whole-percepts**. So `symbol_dim` on WS is misplaced.
3. **`SymbolSpace` / `SymbolicSubSpace` / `symbolize` / `MetaSymbol`** — genuinely
   **SYMBOLS** (the symbol tower + its ops). Keep "symbol" here.

## Current → target term map

| Current | Target | Where |
|---|---|---|
| "symbol table", `_sym_parts`/`_sym_wholes`/`_sym_relate_idx`/`_sym_next`/`_sym_identity`/`_sym_raised` | **Concept codebook**, `_concept_*` | ConceptualSpace (Spaces.py:13343+) |
| `reify_relation`, `new_symbol`, `synthesize_higher_order`, `symbol_is_identity`, `resolve_identities` | concept-minting verbs (`reify_concept`/`new_concept`/…) | ConceptualSpace |
| `create_word_object_meta` (A=word/B=object/C=meta) | mints a concept (C) + its symbol; keep but document as concept+symbol | ConceptualSpace |
| WholeSpace `symbol_dim`, `nOutputDim=8` emission | the **symbol** (moves to SS); WS content = **whole-percepts** | WholeSpace / configs |
| PartSpace "atoms"/"parts"; WholeSpace "wholes"/"properties" | **part-percepts / whole-percepts** (both *percepts*) | docstrings, docs |
| `SymbolicSubSpace` | SS grammar/STM carrier — DECISION: keep, or rename `SymbolSubSpace` (drop "ic") to match `SymbolSpace` | Language.py:9122 |
| `symbolize`, `MetaSymbol`, symbol codebook | keep (true symbols) | Language.py |
| grammar tiers P/C/S | percept-tier / concept-tier / symbol-tier (ties into the tier collapse, Stage 5) | grammar XML, Language.py |

## Scope of the renames

- **Docs (the doc pass, soon):** Mereology.md, Spaces.md, Architecture.md,
  Language.md, STM.md, Reasoning.md — use percept/concept/symbol consistently;
  retire "symbol table" → "Concept codebook".
- **Code identifiers (the careful pass, later):** the big one is CS `_sym_*` →
  `_concept_*` + the concept-minting verbs — many call sites, do it like the
  `SymbolicSpace→SymbolSpace` rename (word-bounded substring sweep + suite-green
  gate). Defer until after the structural refactor (Slice B) so it's one churn.
- **Configs:** any `<symbol*>` knobs that actually mean concept/percept.

## Ordering

1. **NOW — this plan.**
2. **SOON — doc pass:** rename in the live docs (percept/concept/symbol).
3. **LATER — code pass:** `_sym_*`→`_concept_*` etc., post-Slice-B, suite-gated.

## Resolved decisions (Alec 2026-06-21)

- **`SymbolicSubSpace` → `SymbolSubSpace`** (drop "ic" to match `SymbolSpace`).
  Mechanical rename across code + tests, folded into the code-identifier pass.
- **`_sym_*` → `_concept_*`** (+ the concept-minting verbs `reify_relation`→
  `reify_concept`, `new_symbol`→`new_concept`, `symbol_is_identity`→
  `concept_is_identity`, etc.): YES, full code rename. Large mechanical sweep, done
  post-Slice-B, suite-gated (word-bounded substring, like SymbolicSpace→SymbolSpace).

## Cross-references

- memory `symbolspace-refactor` — the codebook architecture (Part/Whole/Concept/
  Symbol) this names.
- `doc/Mereology.md` — the dual-coded explicit-symbol/implicit-percept design.
- `doc/old/2026-06-21-higher-order-symbolic-composition.md` — the intensional concepts.
