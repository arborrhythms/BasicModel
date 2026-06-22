# Higher-Order Symbolic Composition — plan for the next iteration

**Status:** PLAN (2026-06-21). Deferred to a future iteration; the SymbolSpace
refactor that this builds on is landing now (see the memory note
`symbolspace-refactor` and `doc/Mereology.md` §"Explicit symbols ⟷ implicit
subsymbolic representations"). This document captures the *intensional* higher-order
work so the current refactor can stop at the structural cutover.

---

## 1. Where this sits — the two coordinating loops

The SymbolSpace refactor splits the per-stage forward into two loops that the
ConceptualSpace 3-stream bind ties together (PS, WS, SS as peers):

- **Subsymbolic loop — PartSpace + WholeSpace (EXTENSIONAL).** Tall (concept-dim)
  perceptual towers. Parts (bottom-up σ synthesis) and wholes (top-down π analysis)
  are dimensionally-embedded vectors with `.what/.where/.when`. This loop ALREADY
  builds **extensional higher-order percepts** via order-raising (σ pulls many parts
  into a proximal cluster, quantize collapses it onto a higher-order codebook row —
  see Mereology.md "Order-raising"). An extensional higher-order percept = "this
  particular collection of parts, observed."

- **Symbolic loop — SymbolSpace (INTENSIONAL).** A third perceptual tower whose
  codebook holds **symbols**: 0-D (1-D when collected), NOT dimensionally embedded,
  each REFERENCING a concept in the CS Concept codebook. Today this loop runs the
  GRAMMAR (compose/generate dispatch). The *next-iteration* work is to make it build
  **intensional higher-order concepts**.

Both run concurrently during serial processing — words get identified at the same
time the sentence is processed.

**The thesis of this plan:** the symbolic loop builds higher-order concepts
**intensionally** (by rule / definition), and those concepts **correlate with and
structure** the extensional higher-order percepts the subsymbolic loop already
forms. Extension = the observed collection; intension = the rule that generates it.
The two must agree (a concept's intensional definition picks out the same members
its extensional percept collected).

---

## 2. The symbol ⟷ concept correspondence — 1:1 via higher-order collapse (Alec 2026-06-21)

- **Concept codebook (CS).** `_sym_parts`/`_sym_wholes` (Spaces.py): a concept is a
  SINGLE part-whole connection — `[part, whole]`, one part-ref + one whole-ref (PS/WS
  codes by reference); DIACHRONIC.
- **The collapse = mereological order-raising.** A "thing" makes a **MANY:ONE**
  asymmetry — many parts under one whole, or many wholes under one part (the "many"
  detected by co-presence: the same `.where` + `.when`). Collapse it by creating a
  **higher-order mereological part or whole — a SUPERSET**: many-parts-under-one-whole
  → a higher-order **part** (the superset of those parts); many-wholes-under-one-part
  → a higher-order **whole**. As a superset it **subsumes** the members, so they no
  longer appear independently.
- **many:one → one:one → 1:1 concept↔symbol (no lookup).** The superset replaces the
  many:one symbolic relationship with one:one (one higher-order part : one whole, or
  one part : one higher-order whole), so each (collapsed) concept maps to exactly one
  symbol — a direct 1:1 handle, no lookup. *(This IS the existing mereological
  order-raising — `maybe_raise_order` / Mereology.md "Order-raising"; verify the entry
  point at build time. It supersedes the earlier flat-concept-activation LOOKUP idea.)*
- **The same collapse is the universal binding:** co-present parts collapse into the
  **Object**; the co-present **Word + Object** collapse into the meta-concept
  `[Object, Word]` → the **meta-symbol**; recursion up to `symbolicOrder` builds the
  **taxonomy**. `reify_relation` (associating two concepts) is then just collecting
  the corresponding two symbols.

---

## 2b. Meta-concepts and symbols-as-sets — the taxonomy (Alec 2026-06-21)

ONE uniform structure across orders: **a concept is a `[part, whole]` pair.** Order
0: `Concept = [part-percept, whole-percept]`. Higher orders reuse the SAME shape with
concepts in the slots — this is the concrete substrate of §3:

- **Word** and **Object** are each a *concept* (`[part, whole]` of percepts; Object
  spans the ATOM↔UNIVERSE poles), each represented by a **single symbol**.
- A **meta-concept** keeps the structure: `Meta-concept = [Object, Word]` =
  `[part = Object, whole = Word]`. Word (label/category) is the **whole**, Object
  (referent/instance) the **part**, so `Word ⊇ Object` is taxonomic containment. Its
  **meta-symbol** is the single symbol for that meta-concept (Word≡Object; the
  existing `create_word_object_meta`).
- **Recursion up to `symbolicOrder`:** symbols are **stored as parts and wholes** of
  higher-order `[part, whole]` concepts. A symbol-part ∈ a symbol-whole is **set
  membership**; collecting symbols this way yields **mathematical sets** whose nesting
  is a **taxonomy** (symbolic subsumption — classes as discontiguous prototypes — NOT
  the order-0 `.where` percept meronomy; see the meronomy memory).

So §3's "higher-order concept built directly, members = the many" is precisely
**symbols-stored-as-parts/wholes → `[part,whole]` meta-concepts → sets → taxonomy**,
with `symbolicOrder` bounding the recursion. This matches the CS Concept codebook's
`[_sym_parts, _sym_wholes]` (one part-ref + one whole-ref per concept), uniform across
orders. The §3.1 implicit (σ+quantize) vector is the subsymbolic correlate of each set.

### Attention shapes `.where` from CS (Alec 2026-06-21)

PS, WS, and SS are all `PerceptualSpace`s: each has a **`.what`** (its codebook) and
a **`.where`**. **Attention ranges over all three `.what`s, and CS is the natural
home to shape the `.where`** — CS binds the three towers (it alone sees PS/WS/SS), so
the attention/`.where`-producer lives in CS, attends over the towers' `.what`, and
writes each tower's `.where` (this is the home for the existing CS `.where`-passback,
`_passback_scope_*` / `passback_action`). This is the concrete form of the "map
attention to PS/WS/SS" work: **home = CS, targets = the three towers' `.what` → `.where`.**

---

## 3. Higher-order intensional composition — the build

### 3.1 Order-raising, on the symbolic side

The subsymbolic order-raising (Mereology.md) fires when a SIDE over-collects (too
many parts/wholes under a **common symbol** — NOT a common `.where`, which varies).
The symbolic-side analogue:

1. **Trigger:** many concepts sharing structure (e.g. many parts under one whole
   via a common symbol) — the over-collection signal, keyed on count-under-a-
   common-symbol.
2. **Build the higher-order concept DIRECTLY** (index-value, on the over-collected
   side of the higher-order Concept table): a new concept whose members are the many
   (`('sym', id)` self-references), tying the towers as before. Symbols are built
   directly, not folded.
3. **Train the implicit subsymbolic rep:** σ pulls the many constituents' vectors
   into a proximal cluster (invertible, lossless), then **quantization** collapses
   the cluster onto ONE code in the **higher-order codebook** (the lossy abstraction;
   the codebook *is* the mereological structure, so the snap aligns the vector with
   it). σ is the LOSS TARGET (VQ commitment/codebook loss); the abstraction loss is
   localized at the quantizer.

This is the **dual-coded core**: the explicit symbol INDEXES/NAMES the implicit
quantized vector; the two co-evolve and must agree (a directly-built symbol-index
paired with a quantized vector that is a valid point in the higher-order codebook).

### 3.2 Intensional definition (the new part)

The intensional move beyond the extensional percept:
- An extensional higher-order percept is the *collection* the subsymbolic loop
  observed. An intensional higher-order concept is the *rule* (the symbolic
  composition) that generates members — `lift`/`lower`/`intersection`/`union` etc.
  applied at the concept level.
- The symbolic loop composes concepts (via the grammar ops, now on SymbolSpace) to
  define higher-order concepts by INTENSION; these are then required to **correlate
  with** the extensional higher-order percepts (a correspondence loss / consistency
  check), so the rule-defined class matches the observed class.
- Verification: an intensional concept can be checked at order-0 against an episodic
  exemplar store (the NOT-YET-BUILT episodic LTM noted in the meronomy memory) —
  does the rule's extension match observed members?

---

## 4. Build order (next iteration)

Prereqs (landing in the current refactor): SymbolSpace(PerceptualSpace) as a 3rd
tower; the 3-stream peer bind (PS/WS/SS); WS tall; SS the symbol tower (low-dim
cutover); Concept codebook in CS; grammar dispatch on SymbolSpace.

1. **Symbol codebook = as-many-symbols-as-concepts** + the flat concept-activation
   CS↔SS loop (replace the low-dim cutover).
2. **Symbolic order-raising** + the σ+quantize training of the higher-order
   subsymbolic rep (the dual-coded coupling), keyed on common-symbol over-collection.
3. **Intensional definition + correlation loss** tying intensional higher-order
   concepts to the extensional higher-order percepts.
4. **`reify_relation` via collected symbols** (associate two concepts by collecting
   their symbols) as the live relation-formation path.
5. **Episodic exemplar LTM** (optional) for order-0 verification of intensions.

---

## 4b. The `conceptualize()` redesign — build spec (Alec 2026-06-21)

Supersedes the `SymbolSpace.forward_symbol` hook (which violated "CS couples to SS
only via forward/reverse" — it was model-invoked and reached into WholeSpace).

**The duality (corpus callosum):**
- `SS.forward(snap)` / `reverse(snap)` STAY grammar (SyntacticLayer over the STM) —
  they process **CS.subspace**.
- `CS.forward()` processes **SS.subspace**: it calls `conceptualize()` and reads the
  symbol codebook for the bind leg.

**`conceptualize()` on `SymbolicSubSpace` (called by CS), dispatches existing machinery:**
- 0th → `[part, whole]` concept via `relate`/`reify_relation` (Spaces.py:13387/13402)
- 1st → `[object isa word]` meta-concept via `create_word_object_meta` (Spaces.py:13573;
  A=word `{PS parts}→{WS whole}`, B=object `ATOM→UNIVERSE`, C=reify(A,B))
- 2nd → higher-order object via `synthesize_higher_order` (Spaces.py:13507) /
  `maybe_raise_order` (Spaces.py:16844)
- HOST-SIDE (mutates `_sym_*` dicts) → must be `@torch.compiler.disable`; the
  `symbol_tower` fullgraph-relax (`_host_islands`) already permits the break.

**Data:**
- `SS.subspace.what` = a **tensor symbol codebook** `[nSymbols, symbol_dim]` (symbols
  as vectors, 1:1 with the `_sym_*` concepts) — allocate it (today the SymbolicSubSpace
  `.what` is empty). CS reads it as the SS bind leg (tower-symmetric with PS/WS `.what`).
  *(Resolves a map contradiction: `.what` is the tensor codebook for the bind; the
  `_sym_*` dicts are the host-side concept relations it is 1:1 with.)*
- `CS.subspace` = continuous 8-slot STM; new words pushed to the END, `subspace.index`
  marks live (today: newest-at-slot-0 via `_stm_shift_and_push`, Spaces.py:12516 —
  needs the push-to-end + index change).

**Two PREREQUISITES (genuine design calls, deferred by the map agents):**
1. **percept→code discretization (the crux):** the order machinery needs discrete
   PS/WS codebook ids; CS.forward has continuous percepts. Proposed default: quantize
   the part/whole percept to its codebook index (VQ snap) → the code `reify_relation`
   takes. NEEDS CONFIRMATION.
2. **the data source is dormant:** the autobind that populates concepts doesn't fire
   in `MM_symbol_tower`'s forward (`raised=0`, `_pos_kind` empty). Whatever discretization
   we pick, the config needs a live percept→multi-part path or `conceptualize` is
   data-starved (same gap that left `forward_symbol` dark). Resolve this FIRST so there
   is something to conceptualize.

**Build order:** (a) resolve prereq 2 (autobind/data firing) so concepts exist; (b)
allocate the `SS.subspace.what` symbol codebook (checkpoint: new params); (c) add
`conceptualize(order, …)` (eager island) dispatching 0/1/2 with prereq-1 discretization;
(d) `CS.forward` calls `conceptualize` + reads `SS.subspace.what` → SS bind leg; (e)
STM push-to-end + `subspace.index`; (f) remove `SymbolSpace.forward_symbol` + the
model-loop call; (g) verify off byte-identical + on. KEEP the proven channel (combine
n_streams / bind 3rd leg / gate / config / fullgraph-relax).

**Checkpoint note:** `_sym_*` relation tables are NOT currently persisted (host dicts,
`object.__setattr__`); the symbol codebook `W` will persist as a Parameter, but the
concept relations need a serialization path if they must survive reload.

---

## 4c. Concept storage + discretization (Alec 2026-06-21)

A concept is a **flexible combination of two percepts.** How the two constituents are
stored, by order:

**Order 0 — `[part, whole]`:** both constituents are percepts WITH a `.where` and a
`.when` (spatiotemporally located); one is the part, one the whole. Store them **BY
REFERENCE — never duplicate the codes.** The reference is the code's **index**, or its
**`.where`** (if `.where` is extended to index every entry in the PS/WS/SS codebook).
OPEN: pick the reference mechanism — `index` is uniform across orders; `.where` doubles
as location+reference at order 0 but is 0 at order 1, so it can't be the reference there.

**Order 1 — the meta-concept `[word, object]`:** stores the **word concept** and the
**object concept**; may hold MANY words for one object, or MANY objects for one word.
Both constituents' `.where` and `.when` = **0** (abstract — no spatiotemporal location).

**Order-1 chains — Gallistel's "unitization of behavior":** order-1 metas can CHAIN
into a **vine / degenerate tree = a tail-recursive list over conceptual pairs.** Each
step conjoins `[concept, next-meta]`, where the next-meta holds `[next-concept,
another-meta-or-terminal]`. This is a **causal learning mechanism for indefinitely long
sequences** — a list built from conceptual pairs. (A somewhat inefficient encoding, but
it learns arbitrary-length sequences.)

**Discretization (percept→code) — DATA-DEPENDENT (resolves the §4b open question):**
- **Letters / bytes:** SNAP to the percept codebook — a small, fixed alphabet (few byte
  values), so quantizing to the codebook index is exact enough.
- **Other data:** do NOT snap — PRESERVE the subsymbolic input unsnapped, representing
  it only by its **boundary conditions over parts/wholes** (the `.where` extent/bracket,
  the EndpointSum `[start, end]`). The concept references the continuous percept by its
  boundary, not a quantized code.

**Storage ORDER (Alec 2026-06-21, DECIDED): `[whole, part]` (whole first).** Reads as
`whole <implies> part` / `word <means> object` / `<if> first <then> second`. The
constitutive entailment runs whole→part (a whole implies its parts; the determinate
direction at order 0), and the taxonomic `part→whole` (is-a) is its recoverable
contrapositive dual. The existing relation structure already encodes this (the
`part(A,B)` duality), so it is a presentation convention — `part(A,B)` / `relate(part,
whole)` stay part-first internally; no re-architecture. The
columns align as `whole|part`, `word|object`, `if|then`, `NP|VP` — position 1 is the
whole-role, position 2 the part-role. So: `word`=whole (the type/category ⊇ its
object-instances), `object`=part; `NP`=whole (the entity), `VP`=part (the predicate).
NOTE: the current `create_word_object_meta` puts word in `Parts` / object in `Wholes`
— BACKWARDS from this convention; flip when wiring (word→whole, object→part).

**The existing grammar tree IS nested `[whole, part]` concepts (verified against
full.grammar).** The compose rules are predominantly BINARY — `O1 = op(I1, I2)` for
`lift`(VP), `intersection`(ADJ), `lower`(DET), `union`, `conjunction`, `disjunction`,
`preposition`, `bind`, `isEqual`, `isPart`. Each binary op = a `[whole, part]` concept:
`I1` = the whole (the base/NP-frame operated on), `I2` = the part (the modifier/VP-edit),
`O1` = the reified concept. The tree NESTS them (`O1` → next op's input) = the order-1
chains (the vine). So **the grammar's compose ops ARE concept-formation ops** —
`conceptualize` and the grammar are one machinery. Caveats: unary ops (`not`/`non`/
`exist`/`tense`/`morphology`) are a degenerate single-constituent `[whole,part]`;
symmetric ops (`and`/`or`/`union`) have no inherent whole-vs-part (pick a head conjunct);
and per-op, verify the code puts the base in `I1` (the structural claim holds regardless).

---

## 4d. Phase-2 wiring — DONE (Alec 2026-06-21, "finish all phases")

- **`conceptualize()` = Option B (the relation API), NOT a compiled-CS.forward hook.**
  The autobind (Reset-driven) remains the per-sentence concept-former; `conceptualize`
  is the unified 0/1/2/3 API (`SymbolicSubSpace.conceptualize`) the grammar/compose can
  call. Keeping the autobind avoids re-risking the just-working word-learning path.
- **Bind leg = CS-mediated.** `SymbolSpace.forward_symbol` is RETIRED — SymbolSpace no
  longer reaches WholeSpace. `ConceptualSpace._build_symbol_leg` (an eager island) reads
  the order-raising "meta" codes (via the WS ref), **syncs them onto `SS.subspace.what`**
  (the symbol codebook, the symbols' home), and builds the SS bind leg from it;
  `bind_streams` calls it when 3-stream. Per-stage CS get `_model_symbolSpace` (gated) so
  the sync fires. So "CS.forward processes SS.subspace" = CS reads/writes
  `SS.subspace.what` for the leg. Verified: symbols flow (loss 0.1712), off-path
  byte-identical.
- **Order-1 chains = `conceptualize_chain(concept_ids)`** (order 3): a tail-recursive
  `[whole=current, part=rest]` list (Gallistel unitization) for sequences — ORDERED +
  idempotent. Verified (`test_conceptualize.py`).
- **STM push-to-end:** ALREADY satisfied — the `SymbolicSubSpace` typed STM appends to
  the END (slot=depth); the `ConceptualSpace` idea-STM stays newest-at-slot-0 (flipping
  it would invert 100+ tests, so it's left intact). The live-slot `subspace.index`
  marker is a forward-contract for the (future) index-based grammar dispatch; deferred
  to its consumer to avoid dead code.

---

## 5. Cross-references

- `doc/Mereology.md` — "Order-raising" + "Explicit symbols ⟷ implicit subsymbolic
  representations (dual-coded)" — the extensional side + the dual-coding.
- memory `symbolspace-refactor` — the codebook architecture (Part/Whole/Concept/
  Symbol), the 3-stream bind, the WS-tall + symbol→SS reframe.
- memory `idea-decoder-design`, `truth-ideas-build` — the decode + the relation
  stores the intensional concepts will populate.
- `doc/old/2026-06-20-reasoning.md` — System-2 reasoning that consumes the concept
  relations.
