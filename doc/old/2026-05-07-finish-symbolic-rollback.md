# Finish the Symbolic Space Rollback

## Context

Two prior plans are partially complete:

1. **`1-when-luminosity-was-cheeky-simon.md`** (mereological-regions redesign) was fully
   implemented in code, then rolled back via discussion when its core conflation
   (mereology stored on the codebook vs. logic stored on the activation) was
   identified. All Plan-1 code-level changes are gone from the working tree.
2. **`yes-please-write-the-immutable-thacker.md`** (rollback to the corrected
   architecture) is partially landed. The minimal-impact parts (regional code
   removal, new idempotent-loop test, deprecation fixes) are done. The
   architectural-impact parts (codebook width = `nDim`, intrinsic-snap forward,
   `default_rule` removal, downstream XML audit) all reverted because they
   trigger test cascades that aren't fixable without a deeper, more careful
   pass.

This plan finishes that work. It localizes the cascade, updates the tests that
encode the legacy invariant, and lands the architectural simplification cleanly
across `bin/Spaces.py`, `bin/Language.py`, `data/model.xml`, and the downstream
XMLs.

The post-rollback target architecture (per the conversation that produced
Plan 2) is:

- `SymbolicSpace.subspace.what.W.shape == [V_S, nDim]` — codebook stores the
  symbol's encoding, no leading `[pos, neg]` bivector pinned.
- `SymbolicSpace.nDim == ConceptualSpace.nVectors` — each symbol's content is a
  `V_C`-wide pattern of activation across the named conceptual axes (the
  per-concept "how much of concept_i is this symbol").
- Per-prototype catuskoti bivector `[B, V_S, 2]` lives on
  `subspace.activation`, populated by `Codebook.forward(input, project=True)` —
  the intrinsic snap. The `Codebook.project()` / `project_reverse()` pair is
  the canonical C↔S boundary.
- Bivector activation is uniform across PS / CS / SS at width 2.
- `default_rule` removed from `SpaceSyntacticLayer`; grammar XML is the sole
  source of truth for which rules fire. `data/model.xml`'s default grammar
  block is preserved (already lists `S = sigma(S)`, `C = pi(C)`, `P = sigma(P)`
  and reverses); the default-only fast path is updated to populate
  `current_rules` from the parsed grammar instead of relying on the code-level
  fallback.

## Phase 1 — Diagnose the inter-stage cascade

Step 2 alone (drop `nWhat=2` override; codebook width = `nDim`) raises
`test_mm_xor::test_forward_reverse_reconstructs_input_state` MSE from ~0.003
to ~0.030, consistently across seeds. Direct measurement showed Sigma's own
round-trip on captured intermediate data is exact (MSE 4e-16); the loss is
introduced somewhere in the **plumbing between** `SymbolicSpace[t].forward`
and `ConceptualSpace[t+1].forward` (or symmetrically on reverse) when the
muxedSize override is gone.

**Localize before changing more code.** Instrument the staged pipeline:

1. Wrap every per-stage transform on the C↔S boundary in a hook that captures
   its input and output tensors. Specific candidates to instrument:
   - `Space.forwardEnd` and `Space.reverseEnd` — both reshape using
     `_pre_reshape_output` based on `nOutputDim`. With the override gone, the
     reshape destination changes; the reverseEnd un-reshape may not perfectly
     invert under the new shape.
   - `ButterflyLayer._butterfly_pack` / `_unpack` / `_merge` / `_unmerge` —
     these reshape `[B, N, D]` to `[B, N/2, 2D]` and back; sensitive to D.
   - `vspace.normalize("symbols", target="what")` and the matching
     `normalize("concepts", ...)` calls — range checks emit warnings for
     out-of-range values; behavior may diverge.
   - `wordSpace.forwardSymbols` / `reverseSymbols` — these run when WordSpace
     is wired (always now) and may operate on the demuxed modality slots
     differently when `subspace.nWhat` changes.
   - `SymbolicSpace.resolve()` — needs narrow-event handling for the
     post-rollback codebook (already attempted; just re-add).

2. Bisect: re-apply Step 2 (drop the override), capture per-layer round-trip
   MSE, find the first layer at which the trace diverges from the legacy
   trace.

3. Fix the divergent layer's shape contract OR document why the divergence is
   architecturally acceptable.

Critical files for Phase 1:
- [bin/Spaces.py](basicmodel/bin/Spaces.py) — `Space.forwardEnd` (~line 4906),
  `Space.reverseEnd`, `SymbolicSpace.forward` / `reverse`, `SymbolicSpace.resolve`.
- [bin/Layers.py](basicmodel/bin/Layers.py) — `ButterflyLayer` pack/unpack
  (`_butterfly_pack` ~line 1700, `_butterfly_unpack`, `_butterfly_merge`,
  `_butterfly_unmerge`).
- [bin/Language.py](basicmodel/bin/Language.py) — `WordSpace.forwardSymbols`,
  `WordSpace.reverseSymbols`.

## Phase 2 — Update tests that encode the legacy invariant

The `nWhat == 2` override was introduced 2026-04-18 in commit `85a308e1`
("better mereology") together with assertions in `test_partition_resolve.py`
that pin it. Updating those assertions is required before Step 2 can land.

- **[test/test_partition_resolve.py](basicmodel/test/test_partition_resolve.py)**:
  five tests assert `sym.subspace.nWhat == 2` after `SymbolicSpace.__init__`.
  Update each to assert the natural value
  `sym.subspace.nWhat == sym.nDim` (or equivalently
  `outputShape[1] - nWhere - nWhen`). The 5 tests are
  `test_resolve_balances_pos_and_neg`, `test_resolve_serial_lossless`,
  `test_inside_of_parthood_matches_part_primitive`,
  `test_outside_is_negation_of_inside`,
  `test_symbol_codebook_quantizes_activation_not_what`.

- **[bin/Spaces.py](basicmodel/bin/Spaces.py) `SymbolicSpace.resolve()`**:
  add a graceful no-op for the case where neither the muxed event nor the
  `.what` content is at least 2 columns wide. Implementation:

  ```python
  if event is not None and event.ndim == 3 and event.shape[-1] >= 2:
      bivec = event[..., :2]
  else:
      bivec = (subspace.what.getW()
               if subspace.what is not None else None)
      if (bivec is None or not torch.is_tensor(bivec)
              or bivec.shape[-1] < 2):
          return subspace
  ```

  This unblocks `test_basicmodel.py::TestModelTypeVariants::test_conceptual_order_1`
  which constructs a model with `D_C == 1`.

## Phase 3 — Land Step 2 (codebook width = nDim)

Apply, in order:

1. **[bin/Spaces.py](basicmodel/bin/Spaces.py) `SymbolicSpace.__init__`**:
   drop the override block:
   ```python
   self.subspace.nWhat = 2
   self.subspace.muxedSize = 2 + self.subspace.nWhere + self.subspace.nWhen
   ```
   Replace with a comment block explaining the natural width contract.

2. **`SymbolicSpace._build_what_basis`**: change the codebook row width from
   `2 + self.nWhere + self.nWhen` to `self.nDim`. Where/when ride alongside
   the encoding on the per-batch muxed event tensor; they don't live inside
   the codebook itself.

3. **`SymbolicSpace.resolve()`**: keep the narrow-event fallback from
   Phase 2.

4. After Phase 1's cascade fix is in place, the round-trip should be back
   under 1e-2 MSE on `test_mm_xor`. If not, narrow the cascade fix until it is.

Acceptance for Phase 3:
- `test_partition_resolve.py` (5 tests, with updated assertions): pass.
- `test_basicmodel.py::test_conceptual_order_1`: pass.
- `test_mm_xor.py::test_forward_reverse_reconstructs_input_state`: pass under
  existing 1e-2 threshold.
- All 219 `test_basicmodel.py` tests: pass.
- `test/test_idempotent_loop.py`: still passes (independent of Step 2 — it
  exercises `Codebook.project=True` directly).

## Phase 4 — Land Steps 3, 4, 5, 6 (intrinsic snap + grammar-driven dispatch)

### Step 3 — Intrinsic snap in `SymbolicSpace.forward` / `reverse`

Replace the legacy `Sigma + resolve + codebook quantize` flow with:

- **forward**: after grammar dispatch (which fires `S = sigma(S)` etc. as
  specified by the chart), the codebook lookup is done by
  `subspace.what.forward(act_pre[..., :nWhat], project=True)` returning
  `[B, V_S, 2]`. Store as `subspace.activation`. The muxed event holds the
  post-grammar `act` for downstream consumers.
- **reverse**: read `subspace.activation` (the per-prototype catuskoti
  bivector) and decode via `subspace.what.reverse(bivec, project=True)` —
  the cached SVD pseudo-inverse. Then grammar dispatch (generate direction)
  applies whatever S-tier reverse rules the chart selected.

Critical: the per-prototype shape `[B, V_S, 2]` matches per-slot shape only
when `V_S == N` (the slot count). For configs where they differ, decide
whether to take the diagonal (slot i matches symbol i) or the full
`[B, N, V_S, 2]` outer product. The V_S = V_C alignment in MM_xor / XOR_*
makes this trivial.

### Step 4 — Remove `default_rule` from `SpaceSyntacticLayer`

- **[bin/Language.py](basicmodel/bin/Language.py) `SpaceSyntacticLayer.__init__`**:
  drop the `default_rule` parameter and `self.default_rule` attribute.
- **`SpaceSyntacticLayer._next_rule_name`**: when no chart rule is available,
  return `None` (no fallback). The forward / reverse callers already handle
  `None` as a no-op.
- **`SpaceSyntacticLayer.reverse`**: where it currently checks
  `rule_name == self.default_rule` to decide whether to use
  `_pi_reverse` / `_sigma_reverse`, replace with
  `rule_name == 'pi'` / `rule_name == 'sigma'` directly (the two-pass
  ergodic adapter should still apply for these specific rules whenever they
  fire).
- **`build_space_syntactic_layer`**: drop the `default_rule` parameter.
- **`WordSpace._build_builtin_layers_for_tier`**: drop all
  `default_rule = ...` assignments at the P/C/S branches; drop the
  `default_rule=default_rule` argument to the
  `build_space_syntactic_layer` call.
- **`WordSpace.compose` / `WordSpace.generate`**: in the
  `_grammar_is_default_only` and `useGrammar=='none'` fast paths,
  populate `current_rules` (resp. `generate_rules`) with the per-tier rule
  IDs computed from `TheGrammar.rules` (filter by `tier` and by
  `canonical` containing `.reverse` for the generate direction; map back
  to the forward base rule_id so dispatch resolves to a layer that lives
  in `_by_name`). This replaces the old `default_rule` fallback by making
  the grammar XML the sole source of truth without giving up the
  no-chart-inside-pass speedup.

### Step 5 — Update `data/model.xml` grammar comment

Edit the comment block inside `<grammar>` (around lines 147–165) to reflect
that `default_rule` no longer exists; the grammar XML is the sole source of
truth. The actual rules (`S = sigma(S)`, `C = pi(C)`, `P = sigma(P)`,
reverses) **stay as-is** — they're the grammar's actual instructions.

### Step 6 — Audit downstream XMLs that override grammar

[bin/util.py](basicmodel/bin/util.py) declares
`_NON_MERGING_PATHS = (('WordSpace', 'language', 'grammar'),)` —
any XML that defines its own `<grammar>` block fully replaces the default.
Search `data/*.xml` for such blocks and verify each has the rules it needs.
Configs that previously relied on the `default_rule` code-level fallback for
sigma/pi at any tier need explicit `S = sigma(S)`, `C = pi(C)`, etc.
listed.

XMLs to audit: `MM_xor.xml`, `XOR_grammar.xml`, `XOR_exact.xml`, `MM_5M.xml`,
`LM_5M.xml`, `MM_grammar.xml`, `MM_boolean.xml`, `MM_shamatha.xml`,
`MM_bpe.xml`, `MentalModel.xml`, `RamsifiedModel.xml`, `MM_400M.xml`,
`MM_xor_step3.xml`, `MM_xor_step4.xml`.

## Phase 5 — Resolve the V_S = V_C question across configs

Conversation produced two readings of the architecture:

- **(A) Same dim**: `SymbolicSpace.nDim == ConceptualSpace.nDim` (= D_C).
  Each symbol is a point in the ambient concept space. Sigma is square
  invertible. This is what most current XMLs have.
- **(B) Coefficient basis**: `SymbolicSpace.nDim == ConceptualSpace.nVectors`
  (= V_C). Each symbol's content is a coefficient vector over the named
  concepts. Sigma is rectangular (`D_C → V_C`), reverse is pseudo-inverse,
  round-trip is lossy. This is what was discussed at the end of the prior
  session and what `data/MM_xor.xml`'s edits were heading toward.

**Decide** which interpretation lands. Open question for the implementer:

- If (A): no XML changes needed. Proceed.
- If (B): update each model XML to match. The butterfly volume constraint
  (`nPercepts × percept_width == nSymbols × symbol_width`) typically forces
  `nSymbols × V_C == nPercepts × D_C`, i.e. `nSymbols == nPercepts × D_C / V_C`.
  For MM_xor (8 percepts, D_C=10, V_C=8): `nSymbols = 10`. Audit each XML for
  this constraint. The Sigma layer's pseudo-inverse round-trip will
  dominate `test_mm_xor`'s reconstruction MSE — likely needs threshold
  relaxation in that test.

Recommend (A) to land first (no XML / test threshold changes), then revisit
(B) as a follow-on once the architectural simplification is stable.

## Phase 6 — Documentation updates

After the architectural simplification lands, the project's `doc/*.md` files
need to reflect the new shape. The conversation that produced this plan
clarified several things that aren't captured anywhere in the current docs;
add them.

### 6.1 Modes of operation — short summaries to add to `doc/Architecture.md`

The architecture has three orthogonal mode dimensions (set independently in
the model XML):

**Butterfly mode** (`<useButterflies>true</useButterflies>`):

  Pairwise sigma/pi with N-halving across `<conceptualOrder>` stages.
  Each per-stage `ConceptualSpace`/`SymbolicSpace` instance is built with a
  butterfly-mode SigmaLayer/PiLayer that packs `[B, N, D]` to
  `[B, N/2, 2*D]` before the inner LDU and unpacks after. Slot count
  halves per stage; per-slot dim doubles; total per-stage volume preserved.
  Validated by `nPercepts × state_dim == nSymbols × symbol_width` at
  `ModelFactory.validate_config`. Used by MM_xor, MM_5M, MM_400M. Without
  butterflies, every stage shares the same shape and no halving happens
  (the "plain" / `useGrammar=all` paths).

  Reference: [bin/Models.py](basicmodel/bin/Models.py) `_build_staged_pipeline`
  butterfly branch (~line 4289+); [bin/Layers.py](basicmodel/bin/Layers.py)
  `ButterflyLayer._butterfly_pack` / `_butterfly_unpack` /
  `_butterfly_merge` / `_butterfly_unmerge`.

**Serial mode** (`BASICMODEL_DEVICE` / runtime flag `serial_mode`):

  A runtime fast path for streaming / autoregressive contexts.
  `PerceptualSpace`/`ConceptualSpace` may use the slide-and-recompute path
  where the previous step's per-cell warm cache (kept on
  `subspace.serial_cache`) is reused for cells that haven't shifted.
  Independent of butterfly / parallel mode — gated only by the runtime
  flag. The cache is keyed on the owner-Space id, cleared on hard `Reset`,
  rebuilt cheaply on the next forward.

  Reference: `Space.serial_mode` flag (set by `BaseModel.create_from_config`),
  `SubSpace.serial_cache` dict, and the `test_serial_mode_*.py` tests.

**Parallel vs Grammar mode**
(`<architecture><mode>grammar|parallel</mode>`):

  Mutually exclusive Phase-1 modes (per the
  `2026-05-05-subsymbolic-knowing-handoff` plan):
  - **`grammar` mode**: `SymbolicSpace` is active; `SubsymbolicSpace.held_at_zero = True`.
    The symbolic re-entrant loop fires; the subsymbolic event tensor is
    held at zero and contributes nothing to the next conceptual order's
    combined input.
  - **`parallel` mode**: `SubsymbolicSpace` is active; `SymbolicSpace.held_at_zero = True`.
    The subsymbolic (imagistic / felt-sense) re-entrant loop fires; the
    symbolic event tensor is zeroed.

  Both modes wire a `SubsymbolicSpace` parallel to `SymbolicSpace`; only
  one is "running" per pass. The other's event tensor is summed
  elementwise into the next conceptual order's combined input (zeros are
  identity under the additive sum, so the held-at-zero side contributes
  nothing).

  Reference: `held_at_zero` attribute on `SymbolicSpace` and
  `SubsymbolicSpace`; `BaseModel.__init__` mode-dispatch around the
  subsymbolic enable check.

**Subsymbolic enabled** (`<subsymbolicEnabled>true</subsymbolicEnabled>`,
default false):

  Independent of the mode flag — controls whether `SubsymbolicSpace` is
  *constructed* at all. When false, only `SymbolicSpace` is built; no
  parallel re-entrant loop; the per-stage ConceptualSpace's PiLayer is
  not widened. When true, both spaces exist and the mode flag
  determines which one's event contributes to the next conceptual order.

### 6.2 Geometry of conceptual and symbolic space — short summary to add to `doc/Spaces.md`

The current `doc/Spaces.md` describes the legacy `nWhat=2` codebook
contract. After this rollback, the geometry is:

**Conceptual space — continuous, mereological** (unchanged):

  - Codebook `ConceptualSpace.subspace.what.W`: `[V_C, D_C]` unit-norm
    rows. Each row is a *named direction* in the D_C-dim ambient belief
    space; concepts are points on the unit (D_C − 1)-sphere.
  - Activation per slot: signed projection onto each named direction.
    Magnitude carries belief certainty (+1 affirmed, 0 unknown, −1
    denied; intermediate values are partial belief with sign).
  - `use_dot_product = True` → retrieval via single matmul
    `argmax_i (x · c_i)`; codebook unit-norm preserved by EMA.
  - Pi (multiplicative AND-fold) factors percepts into concepts.
  - Mereology measures (Area, Luminosity, Contiguous, Continuous,
    Peaceful) operate here — concept space is where extents and overlaps
    live geometrically.

**Symbolic space — discrete, logical** (post-rollback):

  - Codebook `SymbolicSpace.subspace.what.W`: `[V_S, nDim]` rows. Per the
    target architecture, `nDim == V_C` (i.e. each row is a coefficient
    vector over the named concepts: "how much of concept_i is this
    symbol?"). One symbol per concept under V_S = V_C alignment, but the
    sizes are independent in principle.
  - The codebook is *not* unit-norm — symbol prototypes are free
    coefficient patterns. `use_dot_product = False` → retrieval via
    Euclidean L2 (cached-norm matmul).
  - Per-prototype catuskoti bivector `[B, V_S, 2]` lives on
    `subspace.activation`, NOT in the codebook. Populated by
    `Codebook.forward(input, project=True)` — the intrinsic snap:
    ```
    pos[b, n] = sum_v relu(input[b, v] · W[n])
    neg[b, n] = sum_v relu(-input[b, v] · W[n])
    ```
  - The matching decode `Codebook.reverse(bivec, project=True)` is the
    cached SVD pseudo-inverse. C → S → C round-trip = projection onto
    span(W); idempotent on the codebook's row space after one cycle.

**Activation — universal catuskoti bivector** (post-rollback):

  Width 2 across PS / CS / SS via `ActiveEncoding.nDim == 2`. Encodes
  the four-valued truth lattice (catuskoti / *catuṣkoṭi*):

  | State | `[pos, neg]` | Reading |
  |---|---|---|
  | TRUE (asti) | `[1, 0]` | full affirmation |
  | FALSE (nasti) | `[0, 1]` | full negation |
  | BOTH (ubhaya) | `[1, 1]` | first-class contradiction (paraconsistent) |
  | NEITHER (anubhaya) | `[0, 0]` | unknown / off-lattice |

  Logic operations:
  - NOT: pole swap `[pos, neg] → [neg, pos]` (NotLayer).
  - AND / Intersection: `[min(p_a, p_b), max(n_a, n_b)]`.
  - OR / Union: `[max(p_a, p_b), min(n_a, n_b)]`.
  - De Morgan duality holds.

  Cross-reference: [doc/Philosophy.md](basicmodel/doc/Philosophy.md)
  for the catuskoti mapping.

**The C↔S boundary as categorization**:

  Calling `SymbolicSpace.forward` IS the act of *naming* — projecting
  the incoming concept activation onto the named codebook lattice.
  The snap loss IS the categorization: a clean prototype match yields a
  TRUE-corner activation; a noisy near-prototype match degrades the
  TRUE pole; an off-lattice input collapses to NEITHER. The decode is
  not lossless reconstruction — it's projection onto the codebook
  subspace, which is what categorization means architecturally.

  Grammar (NOT, AND, OR, lift, lower, ...) operates on the bivector
  activation between snap calls. The dialectic loop = snap (synthesis)
  → grammar (logic) → decode (analysis) → next pass.

### 6.3 Files to edit

- `doc/Architecture.md` — add the modes-of-operation summaries (§6.1).
- `doc/Spaces.md` — replace the legacy `nWhat=2` codebook description
  (currently around the "Codebook shape" section) with the post-rollback
  geometry (§6.2). Keep the existing
  "ConceptualSpace concepts are *named directions*" prose as-is — it's
  unchanged.
- `doc/Logic.md` — confirm the catuskoti / De Morgan content reflects the
  activation-as-bivector model. May already be accurate; verify and
  extend with the snap-as-categorization note from §6.2.
- `doc/Philosophy.md` — confirm the four-corner table aligns with
  the post-rollback activation semantics. Likely accurate as-is; no
  change expected.
- `doc/Mereology.md` — confirm the Area / Luminosity / Contiguous /
  Continuous / Peaceful family reads concept activations directly (the
  conversation flagged that mereology belongs at the conceptual tier,
  not on the symbolic codebook). Update if the doc currently describes
  measures running over the SymbolicSpace activation.

Acceptance for Phase 6:
- The five doc files above accurately describe the post-rollback
  architecture as it stands in code.
- The intrinsic-snap semantic is documented in `doc/Spaces.md`.
- Cross-references between docs are consistent (e.g. mereology in
  `doc/Spaces.md` points to `doc/Mereology.md`; logic in
  `doc/Spaces.md` points to `doc/Logic.md` and
  `doc/Philosophy.md`).

## Phase 7 — Optional follow-on (deferred)

Out of scope for landing the rollback, but related future work surfaced in
the conversation:

- **STE soft surrogate**: wrap the snap with `ste_answer(soft, hard)` so
  gradients flow through a Gaussian overlap surrogate when codebook
  prototypes are far from the incoming activation. Currently the snap is
  differentiable through `clamp(min=0)` and division — sparse but not zero
  gradient.
- **Distributed K-identity adapter**: replace the per-slot integer winner
  with a `[V_S, K]` linear projection (K ≪ V_S) so the symbol output is
  a small continuous signature instead of a one-hot. Same operation as
  ConceptualSpace's `use_dot_product = True` retrieval, applied to symbols.
- **OMP-style residual decomposition**: when `top_k > 1`, pick winner #1,
  subtract its contribution from incoming, pick winner #2 from residual,
  etc. Handles the non-orthogonal-codebook case cleanly.
- **TruthLayer empirical-variance accumulation**: aggregate per-symbol
  spread over observations into the existing `record_batch` infrastructure;
  recoverable downstream as the symbol's "extent" without storing radii in
  the codebook.
- **Sparse / factored codebook**: store `[K, D_C] × [V_S, K]` (sparse
  selection of factors) instead of dense `[V_S, D_C]`. Compresses for very
  large vocabularies.

## Critical files

- [bin/Spaces.py](basicmodel/bin/Spaces.py)
  - `SymbolicSpace.__init__` — drop `nWhat=2` and `muxedSize=2+obj` overrides
    (Phase 3).
  - `SymbolicSpace._build_what_basis` — codebook width = `self.nDim`
    (Phase 3).
  - `SymbolicSpace.forward` / `reverse` — replace legacy flow with
    intrinsic-snap (Phase 4 Step 3).
  - `SymbolicSpace.resolve` — narrow-event fallback (Phase 2).
  - `Space.forwardEnd` / `reverseEnd` — likely cascade source (Phase 1).
- [bin/Language.py](basicmodel/bin/Language.py)
  - `SpaceSyntacticLayer.__init__` / `_next_rule_name` / `reverse` — drop
    `default_rule` (Phase 4 Step 4).
  - `WordSpace.compose` / `WordSpace.generate` — populate `current_rules` /
    `generate_rules` from grammar XML in default-only fast path
    (Phase 4 Step 4).
  - `WordSpace._build_builtin_layers_for_tier` — drop `default_rule`
    assignments (Phase 4 Step 4).
  - `WordSpace.forwardSymbols` / `reverseSymbols` — likely cascade source
    (Phase 1).
- [bin/Layers.py](basicmodel/bin/Layers.py)
  - `ButterflyLayer._butterfly_pack` / `_unpack` / `_merge` / `_unmerge` —
    cascade suspect (Phase 1).
  - `Codebook.forward(project=True)` / `Codebook.reverse(project=True)` —
    already in place; consumed by intrinsic-snap forward/reverse (Phase 4
    Step 3).
- [data/model.xml](basicmodel/data/model.xml) — grammar comment update
  (Phase 4 Step 5).
- [data/MM_xor.xml](basicmodel/data/MM_xor.xml),
  [data/MM_5M.xml](basicmodel/data/MM_5M.xml),
  [data/XOR_grammar.xml](basicmodel/data/XOR_grammar.xml),
  [data/XOR_exact.xml](basicmodel/data/XOR_exact.xml),
  others — audit grammar overrides (Phase 4 Step 6).
- [bin/util.py](basicmodel/bin/util.py) — `_NON_MERGING_PATHS` lists the XML
  paths whose `<grammar>` block fully replaces the default. Reference for
  Phase 4 Step 6.
- [test/test_partition_resolve.py](basicmodel/test/test_partition_resolve.py)
  — 5 invariant assertions on `nWhat == 2` (Phase 2).
- [test/test_idempotent_loop.py](basicmodel/test/test_idempotent_loop.py) —
  already in place; should continue to pass.
- [test/test_mm_xor.py](basicmodel/test/test_mm_xor.py) — round-trip
  reconstruction is the canonical end-to-end check.
- [test/test_basicmodel.py](basicmodel/test/test_basicmodel.py) —
  `test_conceptual_order_1` exercises `D_C=1`, requires the resolve
  narrow-event fallback (Phase 2).

## Reused functions / utilities

- `Codebook.forward(input, project=True)` ([bin/Spaces.py:1603](basicmodel/bin/Spaces.py:1603))
  — the snap. Returns `[B, V_S, 2]` per-prototype catuskoti bivector via
  the cached signed-projection accumulator.
- `Codebook.reverse(bivec, project=True)`
  ([bin/Spaces.py:1713](basicmodel/bin/Spaces.py:1713)) — the decode.
  Cached SVD pseudo-inverse; recovers the input modulo span(W).
- `Codebook.project(input)` / `Codebook.project_reverse(bivec)` — the
  underlying primitives both above wrap.
- Existing grammar dispatch via `SyntacticLayer` / `SpaceSyntacticLayer` /
  `chart.compose` — no new dispatch infrastructure needed.

## Verification

After all phases are complete, run:

```bash
# 1. The new idempotent-loop test (independent of SymbolicSpace; should
#    keep passing).
basicmodel/.venv/bin/python -m pytest test/test_idempotent_loop.py -v

# 2. The tests that encoded the nWhat=2 invariant (Phase 2 updated them).
basicmodel/.venv/bin/python -m pytest test/test_partition_resolve.py -v

# 3. The narrow-D_C edge case.
basicmodel/.venv/bin/python -m pytest \
    test/test_basicmodel.py::TestModelTypeVariants::test_conceptual_order_1 -v

# 4. Catuskoti / negation / mereology — should be unaffected
#    (activation shape and semantics are preserved).
basicmodel/.venv/bin/python -m pytest \
    test/test_quaternary_corners.py \
    test/test_negation_layer.py \
    test/test_grammar_binary_ops.py \
    test/test_resolve_luminosity.py \
    test/test_mereology.py -v

# 5. Pi/Sigma ownership (sigma still constructed, dispatchable via grammar).
basicmodel/.venv/bin/python -m pytest test/test_pi_sigma_ownership.py -v

# 6. End-to-end MM_xor — the canonical round-trip integration check
#    (test_forward_reverse_reconstructs_input_state at MSE < 1e-2 was the
#    hardest blocker for Step 2; should pass after Phase 1 cascade fix).
basicmodel/.venv/bin/python -m pytest test/test_mm_xor.py -v

# 7. Full BasicModel regression.
basicmodel/.venv/bin/python -m pytest test/test_basicmodel.py -q

# 8. The full project test suite via the Makefile target the user prefers.
make test
```

### Acceptance criteria

1. `model.symbolicSpace.subspace.what.W.shape == (V_S, symbol_dim + obj_symbol)`
   — codebook width tracks `nDim`, no forced `2 + obj` override.
2. `model.symbolicSpace.subspace.nWhat == model.symbolicSpace.nDim` — natural
   value preserved through `SymbolicSpace.__init__`.
3. `model.symbolicSpace.sigma` is still constructed (sigma is a
   grammar-dispatched layer in the rule registry, not a code-level default).
4. `SpaceSyntacticLayer` no longer has a `default_rule` parameter.
   `_next_rule_name` returns `None` when no chart rule is available.
   `WordSpace.current_rules` is populated from the grammar XML in the
   default-only fast path.
5. `test_mm_xor.py::test_forward_reverse_reconstructs_input_state`
   passes under the existing 1e-2 threshold (modulo Phase 5 if option B is
   chosen, in which case the threshold relaxation is documented).
6. All `test_partition_resolve.py` tests pass with the updated assertions.
7. `test/test_idempotent_loop.py` still passes (4/4) — its semantics are
   unchanged.
8. `make test` passes the full project suite.

## Notes on what stays out of scope

- The Phase 7 follow-ons (STE, K-identity adapter, OMP, TruthLayer
  empirical variance, sparse codebook) are deferred. They build on the
  intrinsic-snap path but don't block the rollback.
- The user's V_S=V_C / V_C=SS.nDim swap (Phase 5 option B) is a separate
  architectural decision. Recommend landing option A first, decide on B
  after.
- `test_mm_xor` warmup-via-training experiments (run a few SGD iterations
  in `setUpClass`) were tried and abandoned — training optimizes XOR output
  + commitment loss, not round-trip reconstruction, so it didn't help.
  With Phase 1 cascade fix, the warmup shouldn't be needed.
