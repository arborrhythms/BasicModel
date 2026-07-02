# Two-Phase Loops + SparseLayer Relation Consolidation — Implementation Plan

> **STATUS: APPROVED 2026-07-02 (Alec) — ready for execution, begin at Phase
> P1.** Both flagged judgment calls are CONFIRMED: (1) the "relation table
> entirely within SparseLayer" interpretation = per-order layer ownership
> (role-tagged COO + ordered constituent store + transitions + store-level
> lifecycle) with a shared `ConceptAllocator` in the same module; (2) the
> JOINT/sentence concept is the ordered Gallistel CHAIN (`[whole=current,
> part=rest]` links, sequence-capable), replacing the flat combination built
> 2026-07-02 morning.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`)
> syntax. This plan supersedes the forward-transform parts of
> [2026-07-02-sparse-layer-conceptual-embedding.md](2026-07-02-sparse-layer-conceptual-embedding.md)
> (whose SparseLayer substrate, poles, pruning, Hebbian, and recon/render fixes
> all carry forward).

**Goal:** Restructure the conceptual forward into two phases -- a purely
continuous subsymbolic PS/WS$\leftrightarrow$CS pump, then a single late
cutover snap to the order-0 conceptual codebook followed by the quantized
SS$\leftrightarrow$CS symbolic loop -- with the SparseLayer made symbolic-only
and made the single owner of BOTH readings of concept structure: the weighted
role-tagged COO and the discrete ordered relation table.

**Architecture:** doc/Architecture.md "Cognitive grounding" (dense-perceptual
vs sparse-symbolic; CLS) and doc/BasicModel.md "Order, extension, and
intension" are the rationale. doc/old/2026-06-21-higher-order-symbolic-
composition.md sec 4c is the relation-table storage spec this plan wires.

---

## Ground rules (unchanged from the prior plan; override any defaults)

- NO git writes by the implementer (user commits). Fail loud on divergence.
- Code comments are one-liners; architecture goes in doc/*.md (LaTeX math).
- Targeted pytest node IDs while iterating; `make test` is the FINAL gate.
- **Byte-identity gate:** all `symbolicOrder=0` configs stay bit-for-bit
  (Phases P1-P3 are gated on `_sparse_active()` / new knobs); P4's new
  parameters are config-gated AND RNG-neutral (save/restore the RNG around
  construction, the `similarity_codebook` idiom) so default configs keep
  their init streams.
- Repro harnesses drive `m.runEpoch` (bare forward loops never Reset).

## Locked design decisions (Alec, 2026-07-02 review thread)

1. **Two phases, terminal cutover.** Phase A: `subsymbolicOrder` iterations of
   the PS/WS$\leftrightarrow$CS loop, fully continuous (2-stream bind; NO
   symbol leg, NO snap inside the pump). Phase B: cut over to the
   SS$\leftrightarrow$CS loop. Quantization happens ONLY at the cutover,
   because symbols (0-D) lack the bandwidth to carry subsymbolic content --
   the snap sits exactly at the bandwidth seam, as late as possible.
2. **Order-0 concepts = the snap.** The settled mixed field is snapped to the
   ORDER-0 BLOCK (`caps[0] = N/2` rows) of the conceptual codebook
   (`similarity_codebook`). Read = differentiable normalized-sum presence
   (magnitude-preserving, per the unit-hypercube argument); write = EMA
   (identity/position; codebooks stay EMA-only). Discretization is
   DATA-DEPENDENT (sec 4c): letters/bytes snap; continuous data is referenced
   by boundary (`.where` extent) -- v1 targets text (snap), the boundary path
   is documented future scope.
3. **SparseLayer is symbolic-only.** No percept families at any order; the
   `[PS|WS]` presence blocks and `_n_ps_codes`/`_n_ws_codes` layout retire.
   Order-$k$ ($k \ge 1$) composes lower-order symbol activations only, plus
   the EVERYTHING bias column. `a_0` comes from the snap; `a_k = \tanh(S_k
   [a_0..a_{k-1}] + bias)`.
4. **Role-tagged edges carry the relation order.** Each symbol family's
   columns split into a WHOLE-role block and a PART-role block (+ the bias
   column). The relation table is the DISCRETE-BINARY READING of this store
   (a row with exactly one whole-role and one part-role edge = the ordered
   pair); embedding a pair is exact, discretizing a general row is defined
   only on the binary-ordered subset. Storage order is `[whole, part]`
   (whole first: whole $\Rightarrow$ part / word $\to$ object / if $\to$ then
   / NP|VP -- sec 4c DECIDED). This is what lets the table store time series
   over symbols (Gallistel chains) and non-material implication.
5. **The relation table and its logic live in the SparseLayer class**
   (Alec, this thread). Interpretation (CONFIRMED by Alec): SparseLayer
   instances are per-order, so each per-order layer owns its rows' ordered
   constituent references, idempotency caches, and the embed/discretize
   transitions; a thin shared `ConceptAllocator` (same module) owns global
   concept ids and order derivation. ConceptualSpace keeps only ORCHESTRATION
   (autobind, span knit, statement channel, lifecycle policy) as delegates.
6. **META/JOINT under role-tagging.** The meta is an ORDERED PAIR: slot 1
   (whole-role) = the word symbol, slot 2 (part-role) = the object symbol --
   reconciling sec 4c's `[whole=word, part=object]` with "a combination, not
   a subsumption" (roles are positional slots of an ordered pair, not
   containment claims). FLIP the current create_word_object_meta shape (it
   has word/object as two parts) to the sec-4c convention. The JOINT/sentence
   concept is an ordered CHAIN over the row's word symbols (sequence = the
   Gallistel vine), each link `[whole=current, part=rest]`, bias-bounded.
7. **Feedback to the towers is percepts, not mixes.** The corpus callosum's
   reverse (`unbind`) demuxes: part-stream $\to$ PS (further $\sigma$
   synthesis), whole-stream $\to$ WS (further $\pi$ analysis). The mix goes
   UP (concept), the un-mix goes DOWN (per-tower percepts). Plain unbind in
   v1; residual (concept minus already-settled) is a later refinement.
8. **Distinct $\sigma$/$\pi$ layers per subsymbolic order** (v1, not
   deferred): PS holds `sigma[t]` and WS `pi[t]` for `t in 0..subsymbolicOrder-1`
   (today one reused `self.sigma`/`self.pi`). Depth IS mereological order;
   the stack is the subsymbolic reasoning engine. Config can mark any order
   as NO-OP (identity pass-through). Loop always runs to `subsymbolicOrder`
   (no adaptive early exit in v1); the QE snap-error is read as a per-percept
   settle SIGNAL only (SymbolLearningLayer stays gated).
9. **Convergence target:** the subsymbolic loop, driven by top-down (basic
   level via WS$\to$PS scope) and bottom-up (longest-match) attention,
   settles at a possibly-partial tiling of the input; the snap reads that
   settled field. Cross-tower "participation in concepts exerts force on the
   mereological space" (parts and wholes toward ONE well-defined space) is a
   FUTURE training-dynamics extension, noted not built.
10. **`sparseReplace` retires.** Phase separation makes non-replacement
    structural: sparse content never substitutes subsymbolic content; the
    symbolic phase's outputs feed the SS leg, the head-side losses, and the
    concept table. (The knob was a bridge; remove it with its tests, or keep
    parsing it as an inert deprecation warning -- implementer's choice,
    recorded.)

## Verified anchors (2026-07-02; re-grep at execution, names over lines)

| Piece | Anchor |
|---|---|
| Parallel body pump + SS leg in-loop + bind | `bin/Models.py` `_forward_body` (~`:6528-6812`); SS leg at `:6757`; stamps at `:875-900` |
| `cs.forward` sparse call + `_concept_activations` stamp | `bin/Spaces.py:15900-15920` area (`_sparse_concept_forward` at `:14316`) |
| Sparse families / layout / populate / prune / poles | `bin/Spaces.py:13972-14295` |
| Relation-table core (to move) | `bin/Spaces.py` `_concept_tables` (~`:13690`), `new_concept/add_part/add_whole/concept_parts/concept_wholes/relate/reify_concept/retire_concept` (~`:13700-13780`), `resolve_identities` (~`:13790`), `synthesize_higher_order`, `conceptualize_chain`, `refine_over_collected`, `prune_concept_links`, `create_word_object_meta`, `create_joint_concept`, `assert_concept_relation`, `_hebbian_strengthen` |
| SparseLayer | `bin/Layers.py:4447` region |
| PS `self.sigma` / WS `self.pi` (single, reused) | `bin/Spaces.py:8528/8530` region |
| unbind / bind carrier | `bin/Spaces.py` `bind_streams` (~`:15660`), `unbind` (~`:15750`) |
| SS leg | `bin/Language.py` `forward_concept_to_symbol` (~`:12734`) |
| Snap-error machinery (signal only) | `SymbolLearningLayer` `bin/Layers.py:11080`; WS LBG `:18312/18359`; RadixLayer promotion `:10285` |
| Tests to migrate/extend | `test/test_cs_symbol_table.py`, `test_cs_sparse_weights.py`, `test_sparse_concept_e2e.py`, `test_cs_to_ss_forward.py`, `test_mereology_word_binding.py`, `test_sparse_layer.py` |

---

## Phase P1 — SparseLayer absorbs the relation table (behavior-preserving)

The consolidation refactor, done FIRST while the current suite is green.

**New in `bin/Layers.py`:**

- `class ConceptAllocator`: global concept-id allocation (`new_concept()`),
  id $\to$ order map, retire set, the idempotency caches that are global
  (`relate`/`reify`/`chain`/word-object-meta keys). One per ConceptualSpace,
  shared by its per-order layers.
- On `SparseLayer` (each instance = one ramsified order's symbol family):
  - **Role-tagged columns:** constructor gains
    `roles=("whole", "part")` block sizes; `add_edge(row, col, role=...)`
    resolves the block offset; the bias column is the trailing column of the
    part... (decide: its own 1-wide block; document).
  - **Ordered constituent store:** `row_constituents[local_row] =
    [(role, ref), ...]` in insertion order -- the sec-4c ordered references
    (refs are global concept ids for sym constituents; order-0 refs are
    codebook row indices).
  - **The two readings:** `embed_pair(whole_ref, part_ref, weight=1.0)`
    (exact: two role-tagged edges + the ordered constituent record) and
    `discretize_row(local_row) -> (whole_ref, part_ref) | None` (defined only
    when the row holds exactly one whole-role and one part-role edge).
  - **Lifecycle logic that is store-level:** identity detection (1:1),
    over-collection counts, edge pruning by role, Hebbian strengthen --
    moved from ConceptualSpace, operating on the layer's own store.
- ConceptualSpace keeps the SAME public API (`new_concept`, `add_part`,
  `add_whole`, `relate`, `reify_concept`, `conceptualize_chain`,
  `create_word_object_meta`, `create_joint_concept`,
  `assert_concept_relation`, `resolve_identities`, `refine_over_collected`,
  `prune_concept_links`, `retire_concept`, `concept_parts`,
  `concept_wholes`) as THIN DELEGATES to the allocator + per-order layers, so
  `test_cs_symbol_table.py` / `test_mereology_word_binding.py` keep passing
  with minimal edits. `add_part`/`add_whole` become the role-tagged writes
  (part-role / whole-role) -- the sets-based `_concept_parts/_concept_wholes`
  dicts retire in favor of the layer store (accessors reconstruct the lists).

**Gate:** full targeted sweep green; `symbolicOrder=0` byte-identical
(relation machinery is host-side; no tensor-path change).

## Phase P2 — Symbolic-only SparseLayer + order-0 snap

- Retire percept families, `_percept_split`, `_n_ps_codes/_n_ws_codes`
  stamps and `cs_source_layout`'s PS/WS blocks. `add_concept_weight`'s
  global-column API becomes role-based (`embed`/`add_edge(role=...)`);
  update `_populate_concept_weights` to write sym-refs by role and the
  EVERYTHING bias only.
- **Order-0 snap-read:** `cs_snap_order0(settled_event) -> a_0` -- the
  normalized-sum presence of the settled mixed field against the ORDER-0
  block atoms (`order_slice(0)` rows of `similarity_codebook`), tanh-squashed
  (magnitude-preserving; NOT cosine). EMA write of matched rows toward their
  winning slot contents (`no_grad`; the identity/position trace).
- `cs_forward_content` v2: `a_0` from the snap; for `k >= 1`,
  `a_k = tanh(S_k.forward_linear_roles(a_0..a_{k-1}) + bias)`; decode
  unchanged (`a_k x softplus(atom)`); returns `(content, a_list)` with
  content used ONLY for losses/SS-leg (never substituted -- P3).
- Mint sites re-point: order-0 concepts (the A word-symbols, the knit) now
  RESERVE order-0 codebook rows (allocator maps concept-id $\to$ order-0 row)
  instead of writing PS/WS-column edges; their part/whole DECOMPOSITION lives
  in the PS/WS codebooks + the ordered reference store (sec 4c: store by
  reference, never duplicate codes).
- **Convention flip** (sec 4c): `create_word_object_meta` meta pair =
  `[whole=word-symbol, part=object-symbol]` via `embed_pair`; JOINT =
  `conceptualize_chain` over the row's word symbols (ordered vine), each link
  `[whole=current, part=rest]` + bias. Update the tests that pinned the
  two-parts shape.

**Gate:** targeted suites (rewritten expectations); `symbolicOrder=0`
byte-identical (all of P2 is behind `_sparse_active()`).

## Phase P3 — Two-phase forward body + demux feedback + late activations

- `_forward_body` (parallel, `_sparse_active()` only): the T-stage pump runs
  **2-stream** (no `forward_concept_to_symbol` in the loop; no
  `_sparse_concept_forward` in `cs.forward`); after the pump, ONE cutover
  block on `last_cs`: snap $\to$ symbolic phase (`cs_forward_content` v2)
  $\to$ stamp `_concept_activations` $\to$ build the SS leg once $\to$
  feed the SS/bind consumers and the losses (SBOW C1 unchanged, now on the
  settled slab). `symbolicOrder` counts the SYMBOLIC phase's ramsified depth.
  At `symbolicOrder=0` (incl. the `symbolTower` scaffold path) NOTHING moves
  -- byte-identical.
- **Demux feedback:** inside the pump, the C$\to$P / C$\to$S handoffs carry
  the UNBOUND streams -- `unbind` recovers the part-stream and whole-stream
  from the stage carrier; `_subspaceForPS.set_event(part_stream)` and
  `_subspaceForWS` gets the whole-stream (today: PS is re-fed stage-0 input
  and WS gets the whole mix -- both wrong per the design). Gated
  (`_sparse_active()` or a `<demuxFeedback>` knob if blast radius demands);
  verify against the round-trip/carrier tests.
- `sparseReplace` retires (decision 10).

**Gate:** `MM_sparse_concept.xml` + `MM_20M_xor` sO=1 experiments rerun (the
v5 harness); sO=0 suite byte-identical; fullgraph eager smoke.

## Phase P4 — Distinct sigma/pi stacks + no-op affordance

- PS: `self.sigmas = [SigmaLayer(...) for t in range(subsymbolicOrder)]`; WS:
  `self.pis = [...]` -- pass `t` selects layer `t` (depth = mereological
  order). Config `<subsymbolicStack>true</subsymbolicStack>` gates
  construction (default false initially -- flipping the default shifts every
  config's RNG; do it as a deliberate later cutover). RNG-neutral
  construction (save/restore) so gated-off configs are untouched.
- **No-op affordance:** `<subsymbolicNoop>0,2</subsymbolicNoop>`-style list
  (or per-space attribute) marks orders whose layer is the IDENTITY
  (constructed as a pass-through; still occupies its slot so the loop always
  runs to `subsymbolicOrder`).
- The QE settle-signal readout (per-percept, report-only) is wired as a
  statistic on the pump (no control flow) for later adaptive work.

**Gate:** stack-on driver config trains XOR at least as well as v5;
stack-off = byte-identical everywhere.

## Phase P5 — Docs, configs, verification

- Architecture.md sec A + the cognitive-grounding section: two-phase loop,
  symbolic-only SparseLayer, role-tagged relation consolidation, demux
  feedback, snap-at-the-bandwidth-seam. BasicModel.md cross-refs. todo.md
  MM-20M STATUS updates. Params.md for new/retired knobs.
- Driver configs: `MM_sparse_concept.xml` / the sO=1 variant get
  `<subsymbolicStack>` + capacity sized so order blocks hold the mint load.
- Full verification: targeted sweep, XOR sO=0/sO=1 runs (output correctness;
  recon render observations), `make test`, MODEL_COMPILE=eager smoke.

## Deferred (recorded, not in this plan)

Continuous-data boundary-reference discretization (sec 4c); residual (vs
plain) demux feedback; adaptive QE-driven early exit; cross-tower
participation force (single part/whole space); replace-mode salience
selection (moot unless replacement returns); NL statement wiring; serial
word/object substitution; recon-fidelity design pass (where-band aliasing,
loss magnitude); Hebbian weakening.

---

## EXECUTION NOTES (2026-07-02, executed P1-P5; all gates green)

Suite counts: 2911 (baseline) $\to$ 2917 (post-P2 rewrites) $\to$ 2927
(post-P4, +stack/singleton pins), zero failures at every gate; XOR sO=0
solves to 0.000 (both MM_20M_xor and the MM_sparse_concept sO=0 control).
Load-bearing deviations and mid-execution design refinements:

1. **P1 store keying.** The per-order layer's constituent records are keyed
   by GLOBAL concept id (not local tensor row): capacity overflow must not
   lose records, orders migrate as constituents accrue (`ConceptAllocator.
   settle`), and the duck-typed test stubs need no tensor sizing. The local
   tensor-row map (`assign_row`, capacity read LIVE from `_order_caps`)
   absorbs `_csw_rows` per order; `cs._csw_rows` survives as a merged
   read-only property. `_concept_parts`/`_concept_wholes` survive as
   writable dict-of-set VIEWS (instance attrs, stub-safe).
2. **P2 decisions.** The EVERYTHING bias is its OWN 1-wide role block (the
   flagged judgment call). Order 0 accepts NO edges at all (snap rows only;
   `add_concept_edge(0, ...)` raises). The snap read CLONES the order-0 rows
   (version-counter safety vs the EMA write -- the SS-leg clone lesson).
3. **SINGLETON refinement (Alec, mid-execution).** A 1:1 tie between SYM
   refs is the unit-set/pair STRUCTURE (metas, chain links, singletons) and
   `resolve_identities` never collapses it -- without this the P2 flip
   self-destructs at the first lifecycle pass. `singleton_concept(x)`
   (idempotent, min-support-exempt) and the typed-intersection read-out
   `meta_word_object(C)` landed with it.
4. **P3 demux = `combine.views`, NOT `combine.reverse`.** The exact inverse
   of the bind returns each tower's OWN input (zero information transfer;
   at t=0 the whole-stream is the empty seed forever -- root-caused on the
   sO=1 frozen-loss symptom). The per-tower windows of the MIXED carrier
   are the meaningful un-mix. The in-loop SS leg is suppressed ONLY under
   `_sparse_active()` (the sO=0 symbolTower scaffold keeps its in-loop leg
   -- byte-identity). The cutover uses STAGE 0's cs (the store the autobind
   populates) on the TERMINAL settled event.
5. **P4 gate reading.** Stack-off is byte-identical everywhere (full suite);
   stack-on runs clean and matches the v5 bar (v5 sO=1 was the 'h h'
   collapse; now an undecided plateau). Root-caused for the deferred
   training-dynamics pass: early-stage combine gradients ~100x weaker under
   demux recursion (tanh variance compression), and the snap's slot-mean
   readout input-blind at init on tiny inventories. WS reverse still uses
   the base `pi` (per-pass reverse threading deferred with the stacks'
   training dynamics).
6. **Chain truncation recorded.** At `symbolicOrder=1` a link's edge to the
   previous link (same order) is dropped by stratification -- the discrete
   store keeps the vine exactly; the weighted reading truncates. This is
   the motivating defect for the successor design:
   [2026-07-02-iterated-symbolic-loop.md](2026-07-02-iterated-symbolic-loop.md)
   (DRAFT -- untyped square layer, wave activation, no self-edges, Kripke
   groundedness as the cycle diagnostic).
