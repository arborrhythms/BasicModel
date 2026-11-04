# Item 2.0, basic percepts: the last perceptual layer ties parts and wholes, and order-0 concepts collect basic percepts (Claude, 2026-11-04; spec in code names, with decisions for Alec)

**Status.** Decided by Alec on 2026-11-04 (§1), part of item 2.0 (`todo.md`), beside the [order-zero loop](2026-11-04-item2-0-order-zero-loop.md), whose reverse runs through basic percepts, and the [memory file](2026-11-03-item2-0-behavioral-chains.md). Governed by step-2 §39 ([2026-10-21-step2-one-ltm.md](2026-10-21-step2-one-ltm.md)): within an authorized round Codex reviews the spec, fixes what the code contradicts and continues, documenting each decision as a dated section; only reversing a §1 decision comes back to Alec. The Codex text is held until §5 is decided and joins item 2.0's one Codex text. Code names are from the 2026-11-04 working tree, which carries Codex's uncommitted item 2.5 completion round; the derivation was read from the code and not run.

## 1. The record (Alec, 2026-11-04)

1. "Order 0 concepts are a part of System 1 ... That also means that order zero symbols, which are like percepts in that they still have a spatial interpretation, are a part of system 1."
2. "The last layer of percepts is the basic level, in Eleanor Rosche terms (?). Let's call it that; when a percept has both wholes and parts, it is a basic percept. So this is basically taking the place of the first set of symbols in the current architecture; those symbols remain associated with order-zero concepts. So zero-order concepts are more similar to the existing higher order concepts, whose elements are basic percepts rather than parts and wholes."
3. "Basic percepts are collected by order-zero concepts, which are in turn represented by their order-zero symbols."
4. "There is no bridging unit, basic percepts are purely perceptual, and collected by concepts. Basic percepts do take over the responsibility to tying together parts and wholes that had been done by the first symbolic layer (and they will also implement the centroid positioning of that layer, which makes all percepts well-situated and concrete, and lets all symbols be centroid or whatever mechanism they chose for symbolic space, since symbolic space does not need to have a mereological read at any level)."
5. "An arbitrary number should just be very large, since all parts and wholes will be represented as mereological boundaries within that basis. Pick a number, preallocate, put it in model.xml." The name is "basic". "Parts and wholes collect under basic percepts, which take over the role of zeroth order symbols and exist at the centroid. Concepts then collect those percepts just as all subsequent conceptual layers collect symbols."
6. It "will preserve the perceptual/conceptual back-and-forth that is essential to the basic model."

## 2. What exists, in code names

- **The perceptual stack.** Bytes go to `RadixLayer`, whose `part_groups` and `canonical_parts` nest chunks to any depth; then the percept codebook (`PartSpace.subspace.what`, a `Codebook` with fixed identity rows); then `WordIdentity`: boundary-pair and length atoms, a word row whose code is `cat(form, fill)` with the form the amax over its atoms and nested rows, and whose `inverse_table` entry is the word's bytes. PartSpace composes (`synthesize_word_parts`, the boolean `part_memberships`); WholeSpace cuts the field and holds the properties (`PrimitiveProperties.members`, `evidence_on_counts`), today in `max(1, bindingDepth)` separate copies (§5 D3). The towers hold no learned fold layers. The widths the configurations declare are rewritten at load by `resolve_identity_layout` (`bin/WordIdentity.py` about 17-59): every embedding configuration runs percept and whole events of 360 = 64 + 32 + 256 + 8 (`identityPairDim`, `identityLengthDim`, `propertyDim` and the fixed where/when band), and `data/BasicModel.xml` a concept width of 1256 = 360 + 896, whatever its header says.
- **The first symbolic layer ties parts and wholes today.** An order-0 concept's definition holds literals of both towers: `ConceptualAttentionLayer.features` columns `4*pid + 2*tower + pole`, written by `_populate_concept_weights`'s order-0 branch from `Interpret.lookup_word`'s parts (`identity.admit`) and wholes (`property_rows_for_bytes`). `cs_read_memberships` reads part and whole spans on one event axis, gates every literal by the part membership literal, and folds them under one minimum (`fold_presence(conjunctive=True)`). `MereologicalCodes.derive` places the order-0 form at the identity join of its parts and the property soft meet of its wholes (`_subspace_bounds`). `FormContainment.audit` certifies those premises per tower.
- **The order-0 symbol** is the perceptual face of an order-0 concept, the `[:percept_width]` part of its `similarity_codebook` row; no identifier names it. The documents call it the word form (Orders §9; Architecture, "Two spaces, one index").
- **Gradient.** The parallel field reaches `PrimitiveProperties.members` only through the whole literals; part presence is boolean; the word row's fill is written without gradient (`WordIdentity._fixed`, `refresh_property_fills`).

## 3. The design, in code names

### 3.1 Basic percepts

The last perceptual layer is a preallocated table of basic percepts in perceptual space, `BasicPercepts`, of capacity `basicPerceptCapacity` (§5 D1). A basic percept is a percept with both parts and wholes. Each row records its parts (the PartSpace constituents: identity atoms, nested rows, radix chunks) and its wholes (its WholeSpace properties), and its code sits at their centroid: the identity join of its parts and the property meet of its wholes, which is what `MereologicalCodes._subspace_bounds` computes for order-0 forms today, moved into perception. Every part and whole is a mereological boundary within this one basis. A word that `Interpret.lookup_word` admits becomes one basic percept, the `WordIdentity` row being its precursor. Rows are allocated at construction; a full table raises.

A basic percept's read at a bracket ties its parts and wholes: the presence of its parts and the evidence of its wholes' properties, combined by the minimum that `fold_presence` takes over both towers today, with the complement pole from the property lanes. This is the cross-tower conjunction, moved out of the concept definition into the percept. It is computed live, so the parallel field keeps its gradient to `PrimitiveProperties.members` (the order-zero loop's gate 4); the fill written without gradient goes.

### 3.2 Order-0 concepts collect basic percepts

An order-0 concept's definition is a collection over basic percepts, as a higher-order concept's is over symbols: sigma over pi with one literal kind, `features` column `2*bp + pole`. `_populate_concept_weights`, `add_concept_feature`, `lookup_word` and the other literal writers (`FamiliarTier`, `ConceptLessons`, `_witness_concept_poles`) write basic-percept literals. `cs_read_memberships` reads basic-percept evidence at each bracket and folds it by the existing sigma and pi (`_compose_order0` unchanged, the `working[:n0]` contract kept); the lexical gate and the cross-tower minimum leave the concept read. Order-0 concepts no longer tie parts and wholes, and each is still represented by its order-0 symbol.

### 3.3 Symbols without a mereological read

Basic percepts carry the centroid positioning and the part-whole structure, so symbolic space needs no mereological read at any order. An order-0 symbol is placed as every higher symbol is: at the centroid of what its concept collects (here its basic percepts), or by symbolic space's own placement. `MereologicalCodes.derive` stops deriving order-0 forms from the two towers; `FormContainment.audit`'s premises move to basic percepts.

### 3.4 The reverse

The order-0 reverse of the [loop](2026-11-04-item2-0-order-zero-loop.md) runs back through the layers. An order-0 concept's field evidence goes to its basic percepts by the sigma and pi inverse, as higher-order descent does; a basic percept goes to its parts and their bytes (`WordIdentity.read_parts`, `inverse_table`, `canonical_parts`, `decode_activity`), placed at the field's bracket. No perception spans are needed: this replaces `_reverse_field`'s spreading of whole columns over covering spans and its laying out of part columns by `bytes_for` lengths. Perception stays the target. Forward the order is perception, basic percepts, order-0 concepts; the reverse runs the same layers back, which keeps the perceptual and conceptual back-and-forth.

### 3.5 Names and documents

"Basic percept", after Eleanor Rosch's basic level: the level of categorization at which members share a shape and a configuration of parts, which Tversky and Hemenway (1984) found is where part knowledge concentrates, so the term fits a percept defined by having both parts and a whole. "Order-0 symbol" stays the perceptual face of an order-0 concept. Documents, by exact replacement:
- Orders §1: parts and wholes move to perception; order 0 keeps the Boolean operations. Orders §9: the word as order-0 symbol, now represented over basic percepts.
- Mereology: the join and the meet now live in basic percepts.
- Architecture's "Two spaces, one index" and its order-alternation passage; Spaces.md.
- The item 2.8 spec gains a dated note: its rule that no row has content in both subspaces is superseded for basic percepts, which carry both by definition; its two subspaces stay as their layout.

## 4. Gates (one frozen hash, sized to the change; certificates, no learning claim)

1. **Preallocation.** The table's capacity comes from `model.xml`; the allocation read shows it at construction; nothing resizes; a full table raises.
2. **Admission.** Every admitted word is one basic percept with its parts and wholes, its code at the centroid, laid out in the item 2.8 subspaces.
3. **Definitions.** Order-0 definitions hold only basic-percept literals: a certificate that no part or whole literal column remains in `features`.
4. **The read.** The cross-tower minimum is absent from `cs_read_memberships`; the basic-percept read carries it.
5. **Gradient.** The parallel field reaches `PrimitiveProperties.members` through the basic-percept read.
6. **The reverse.** An order-0 field decodes to bytes through basic percepts with no perception spans; the order-zero loop's two acceptance checks hold.
7. **Containment.** `FormContainment.audit` holds on basic percepts.
8. **One WholeSpace** (§7, D3). One instance on every configuration, every path reading `self.wholeSpace`; `PerceptField` and `SentenceAttention` give a property row one where-address; the word fill and the open read the same `members`. The migration's selection, its raise and its merge of copy 0's host extras are certified on one serial and one parallel checkpoint; the persisted-address certificate is in the receipt.
9. **Unchanged:** the unchanged sets on the same hash, one native lesson smoke, `make test` at the end of item 2.0.

Rules, as in every round: preallocate at construction, never resize; no language named in code, fixtures or tests; no seed selected to pass; existing tests ported with before and after text in the receipt, nothing retired silently.

## 5. Decisions for Alec, with recommendations

- **D1 Capacity.** `basicPerceptCapacity` 262,144 in `data/model.xml`: the corpus fitting target, above the corpus identity audit's 153,655 exact word forms. It is allocated only where the percept store has word identity; lesson configurations may set less. A row of 352 floats (`identityPairDim` + `identityLengthDim` + `propertyDim` in `data/model.xml`: the part and whole subspaces of item 2.8) makes the table about 0.37 GB in float32. Recommended: yes.
- **D2 Concepts that are not word units**: the XOR configurations' property-only concepts and the part-only field candidates (`observe_native_cases`). Recommended: each becomes a collection over the basic percepts that carry the property or the part, so order-0 concepts have one element kind. The alternative keeps property and part literals readable beside basic percepts.
- **D3 One WholeSpace.** Alec: "we should not need copies... what is the motivation for this?" There is none today (§7).

## 6. Where

Item 2.0, on main, after the gradient changes and the memory file and with the order-zero loop, whose reverse runs through basic percepts. Item 1.5 follows and adopts the new `features` layout: at order 0 its fold becomes a gather, and its derive scatters move into perception. Files: `bin/WordIdentity.py`, `bin/Spaces.py` (`cs_read_memberships`, `add_concept_feature`, `_populate_concept_weights`, `_witness_concept_poles`, `PartSpace`), `bin/PerceptField.py`, `bin/MereologicalCodes.py`, `bin/FormContainment.py`, `bin/Interpret.py`, `bin/Layers.py` (`ConceptualAttentionLayer.features`), `bin/FamiliarTier.py`, `bin/ConceptLessons.py`, `bin/Models.py` (`_reverse_field`, `_passback_scope_ps`, the WholeSpace construction loop at about 16624-16806 and its about 61 references), `bin/WhereRegistry.py`, `bin/ModelCandidateAttention.py`, `bin/SentenceAttention.py`, `bin/ConceptLessons.py`, `bin/checkpoint_migrations.py`, `data/model.xml`, `data/model.xsd`, `doc/Params.md`, the about 30 test files that count or key WholeSpace copies, and the documents of §3.5 with `doc/Spaces.md`'s and `data/model.xsd`'s descriptions of the copies.

## 7. Decided (Alec, 2026-11-04)

- **D1** "is fine, but I would prefer powers of two for all constants." The capacity 262,144 is $2^{18}$. Its row of 352 floats is a sum, 64 + 32 + 256, of powers of two. Read here: every constant set by hand is a power of two, and a width made by adding such constants stays a sum. The fixed where/when band of 8 and the exact-length one-hot of 32 (`WordIdentity.py` about 36) make an all-power-of-two width impossible above 16 unless one part is a remainder. *Alec corrects this reading if he wants the sums too.* The audit of every other constant is [2026-11-04-powers-of-two-audit.md](2026-11-04-powers-of-two-audit.md); its pass is in item 2 and its measured values are item 0's decisions.
- **D2** "True": a concept that is not a word unit is a collection over the basic percepts that carry its property or part.
- **D3** "we should not need copies... what is the motivation for this?" Found from the code, the documents, git history and four checkpoints, with an adversarial re-check:
  - *Motivation: none current.* The copies are left from the feed-forward refactor of 2026-04-22 (commit `8be616ad6d`, plan `2026-04-22-pipeline-ff-architecture.md`, since deleted). It replaced per-level views of one ConceptualSpace and SymbolicSpace pair with independent instances per stage, to end autograd retention across forwards and to give each stage its own halved size. WholeSpace is the renamed SymbolicSpace (`cbc6ff855c`, 2026-06-19), and `bindingDepth` the renamed `subsymbolicOrder` (`eb1fbefb5f`, 2026-10-05). Since then WholeSpace holds only properties and is upstream of the symbolic stages, and the halving is gone. No document gives a reason for several property inventories.
  - *Use today.* The serial path runs and trains only the last copy. The parallel path reads, teaches and trains copy 0, and its later passes return a property-neutral zero. In the checkpoints read, the copies not on the path are bit-identical to their initial rows.
  - *Four latent inconsistencies from the split.* The word fill is refreshed from copy 0 while the serial open reads the last copy. The byte cut runs on copy 0. Lessons teach copy 0, which a serial open never reads. One property row has two where-addresses, 27 apart on `data/BasicModel.xml`: `PerceptField` uses the start of the wholes slice, `SentenceAttention` the last copy's offset.
  - *The change.* Exactly one WholeSpace, whatever `bindingDepth` is; `bindingDepth` keeps counting the ConceptualSpace stages and the ramsification slots, and every path reads `self.wholeSpace`.
    - A checkpoint migration keeps the copy whose `members` or `observed` differ from their initial rows (the last on serial checkpoints, copy 0 on parallel ones) and raises if more than one differs. It merges copy 0's host extras (`_standalone_run_bytes`, which shapes the byte cut, `_property_rows_used` and the LBG-split rows) and drops the rest. Only checkpoints written since 2026-10-27 need it; older ones already fail on the WholeSpace width.
    - The where registry's wholes slice shrinks by $(T-1) \times$ `nVectors`, which shifts the symbol and conceptual offsets. Whether any persisted record holds absolute addresses in those slices is certified, and they are migrated if so.
    - `ModelCandidateAttention`'s input width falls (189 inputs on `data/BasicModel.xml`); its weights in old checkpoints are rebuilt and reported.
  - *Not changed.* The ConceptualSpace stages have the same origin, but in the aligned mode they already share one dictionary and they are the binding layers `bindingDepth` counts.

