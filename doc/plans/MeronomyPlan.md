# Meronomy Execution Plan

> **Status: PLAN (rev 2).** Companion to [MeronomySpec.md](MeronomySpec.md)
> (rev. 2026-06-10, review + design session). Baseline: commit `31e69d1`
> ("Finish PS/SS Analysis/Synthesis refactor") — PS owns σ, SS owns π.
> Suite baseline is **zero failures** (device-leak pollution fixed
> 2026-06-10; `conftest` pins the suite to CPU). All `pytest` runs are from
> `basicmodel/`, never the parent repo. Every stage gate is: *full suite
> stays at zero failures + the stage's named tests are green.*

## 0. Strategy

- **New classes, untouched legacy.** Per author direction, the membership
  kernels ship as **`PiLayer2`** and **`SigmaLayer2`** (with a new
  `ContractiveInvertibleLinearLayer` inner layer). Legacy `PiLayer` /
  `SigmaLayer` / `NonNegativeInvertibleLinearLayer` are not modified — no
  flags, no checkpoint rewrites; old checkpoints keep loading on old
  classes; the loader's existing arch-mismatch fail-fast covers the rest.
- **One config knob, dark landing.** A single `<meronomy>` option
  (`off` | `on`, default `off`) selects where the meronymic slots
  (`PS.self.sigma`, `SS.self.pi`) bind `SigmaLayer2`/`PiLayer2` and enables
  the interface factoring and the reference table. Stages land dark; the
  final stage flips the default. Existing model XMLs (`MM_xor.xml` etc.)
  are untouched until cutover, preserving the XOR determinism pins
  (seed-1 + CPU fixtures).
- **Processing contract respected.** Calculations in Spaces
  (`__init__`/`forward`/`reverse`/`reset` only); formulas in `Ops`
  (the existing `Basis` → `Ops` delegation, `Spaces.py:1385–1440`); the
  reference table is model-level data like the codebooks, called only from
  forward paths.

## 1. Stages

### Stage 0 — Laws and constants (pure functions, no wiring)

Files: `bin/Layers.py` (`Ops`), new `test/test_meronomy_laws.py`.

- `Ops.eval_chart(a) = (1+a)/2` — the evaluation map (belief → assumed
  membership; spec §3). `Ops.eval_chart_inv(m) = 2m−1`.
- Constants: `EPS_LOG = 1e−6` (log-floor near `m = 0`),
  `D_MAX_STABLE = 4.0` (default; config-overridable).
- Tests: chart corners exact (`+1 ↦ 1`, `0 ↦ ½`, `−1 ↦ 0`); bijection;
  no `ReLU`-injection law anywhere (the factored interface of Stage 4
  replaces it — guard test asserts the old `max(0, 2m−1)` helper does not
  exist).

### Stage 1 — `ContractiveInvertibleLinearLayer` (spec §4 deltas)

Files: `bin/Layers.py`, new `test/test_contractive_linear.py`.

- New class beside `NonNegativeInvertibleLinearLayer` (`:1772`), not a
  modification of it:
  - diagonal `d = 1 + softplus(raw)` (init `raw ≈ −5` ⇒ `d ≈ 1`);
  - `stable=True` clamps `d` to `[1.0, D_MAX_STABLE]` (legacy clamps to
    `(ε, 1.0]` at `:1822` — opposite regime, untouched);
  - bias `b = −softplus(raw)` (legacy positive path `:958–964` and
    unconstrained invertible path `:1851–1853` untouched);
  - off-diagonals nonnegative via softplus-LDU as in the parent;
  - `blocks=` arg for the binary `2D→D` form: both block-diagonals
    `diag(W_L), diag(W_R) ≥ 1`.
- Tests (§10.10): constraints hold at init and after randomized optimizer
  steps (seed-pinned property test); exact LDU inverse round-trips.

### Stage 2 — `PiLayer2` / `SigmaLayer2` (spec §4)

Files: `bin/Layers.py`, new `test/test_pi2_sigma2_folds.py`.

- `PiLayer2`: forward `exp(W·log(max(m, EPS_LOG)) + b)`, reverse
  `exp(W⁻¹(log z − b))`; inner layer `ContractiveInvertibleLinearLayer`;
  no odds transforms anywhere in the class.
- `SigmaLayer2`: **the De Morgan wrap of the same kernel object** —
  `σ(m) = 1 − π_kernel(1 − m)`; one kernel + one involution.
- Binary folds stay recommender-reversible (existing
  `conjunctionReverse`/`disjunctionReverse` plumbing works on the new
  path).
- Tests: §10.1 order theorems for random admissible weights (`π ≼ min`,
  `σ ≽ max`, elementwise); §10.2 roots/identities/absorbers to ε; §10.4
  De Morgan exactness with the shared kernel; inverse round-trips at the
  `m → 0` corner under `EPS_LOG`.

### Stage 3 — Truth measure: catuṣkoṭi luminosity over codes (spec §3, rev b)

Files: `bin/Layers.py` (`TruthLayer.luminosity`), new
`test/test_truth_luminosity_codes.py`; docstring touch-up on the
`Mereology._luminosity_truth_fold` delegator.

> The original Stage 3 (write-type API, per-symbol κ, conflict
> interrupt) was implemented, rejected on author review (2026-06-10),
> and fully reverted. TruthLayer's storage contract is **unchanged**:
> the catuṣkoṭi accumulator of signed idea-vectors weighted by ±trust.

- Complete the previously stubbed measure: the old
  `luminosity(sym=)` returned `0.0` without a decoder handle, decoded
  every row to concept space (`decode_to_concept` pullback), and folded
  sequentially (order-dependent; the running value was overwritten by
  the last pair). Replace the body with the code-level computation:
  per conceptual dimension, pole coverage `T = relu(rows).max(0)`,
  `F = relu(−rows).max(0)`; `luminosity = mean[(T − F) − min(T, F)]`
  clamped to `[−1, 1]` — total signed area minus sign-conflict regions,
  computed **on the codes** (no pullback; licensed by the §4 weight
  law, accuracy registration-maintained per §5).
- Signature kept (`sym=` accepted and ignored) so `Mereology.Luminosity
  (truth_layer=…)`, `universality`, and `extrapolate` call sites are
  untouched. Paired-index bivector rows are outside the measure's
  domain (orthogonal poles don't oppose); their corner policy stays in
  `tetralemma_balance_penalty`.
- Tests (§10.7 rev): empty store = 0; positive-only store measures
  positive; exact catuṣkoṭi corners per dim (T-only ⇒ +T, F-only ⇒ −F,
  both ⇒ −min, neither ⇒ 0); duplicate truths don't inflate coverage;
  ±trust scales contribution; order-independence under record-order
  permutation; a poisoned `sym.decode_to_concept` is never called;
  range clamp; conflict store measures negative.

### Stage 4 — Interface factoring at the callosum (spec §3)

Files: `bin/Spaces.py` (`ConceptualSpace.bind_streams` `:12255`), new
`test/test_interface_factoring.py`.

- **Parallel mode is the simple mixing matrix** over the `2N` stacked
  slots (`N` = STM size; the existing `[2N,N]` `bind_streams` glue) —
  unchanged structurally; this stage governs *what crosses it*, not its
  shape. (Serial mode bypasses the per-tick fusion entirely; the
  fusion duty migrates into the shift — Stage 7, spec §6 rev c.)
- The PS leg crosses **nameless**: no naming side-channel, no coordinate
  chart. Where the percept path evidences a reference row, the write is
  factored: content → row selection (embedding match against the reference
  half), evidence → activation magnitude `a ∈ [0, +1]`.
- Belief→extent sites use `Ops.eval_chart` (membership-fold operands from
  epistemic scalars).
- Tests (§10.6): no stimulation ⇒ `a = 0`; `a < 0` unreachable end-to-end
  from the percept path; evaluation chart applied exactly once per
  cash-out.

### Stage 5 — Reference-half lookup law (spec §3)

Files: `bin/Spaces.py`, `bin/embed.py` (docstrings/guards only), new
`test/test_reference_lookup_gauge.py`.

- Audit and align lookup sites: reference-half lookups use the
  pole-quotient law (identity = `argmax |q·v|`, polarity = `sign(q·v)` —
  `_pole_aligned_score`, `embed.py:118`); ground/token codebooks keep
  full-vector lookup (`_wrapped_mse_score` / Euclidean paths).
  `ConceptualSpace` storage convention is **unchanged** (unit-norm EMA
  rows, magnitude-as-certainty, `Spaces.py:10874–10882`).
- Gauge fixing at mint: orient `+u` toward agreement with the positive
  referent.
- Tests (§10.3): `u ↦ −u` relabeling observational invariance; certainty
  and polarity survive retrieval (`(a·u)·v = a(u·v)`); token lookup
  unaffected (no NEG-quotient leakage into form codebooks).

### Stage 6 — The word/object binding table (spec §6 rev 2026-06-11, §8)

Files: new `bin/References.py`, new `test/test_reference_table.py`.

> Rev 2026-06-11 supersedes the constructor-cell design (op fields,
> `descend|stop` branch tags, derived order with the `conceptualOrder`
> mint bound, binary-branching census): the table is the **lexicon-like
> binding store**, and all composition lives in the towers.

- **Rows are full bindings only**: `(word: SS row, object: PS row)` —
  a row exists iff both sides are bound; partial rows are rejected at
  the API. There are **no half-bindings anywhere**: unbound words and
  nameless concepts are ordinary tower codes, and bound-ness is never
  stored — it is discovered by query (lookup/search misses ARE the
  "unknown word" / "nameless concept" states).
- **Word-keyed (sorted by word): the word is the reference for the
  object.** `deref(word) → object` is the indexed, cheap direction
  (the serial forward shift's mechanism). `ref(object) → words` has
  **no index by design** — a search of the object side (recommender
  similarity/dominance probe as the primitive); tip-of-the-tongue =
  failed object-side search.
- **Symbols are atomic** (the arbitrariness argument, spec §6): created
  as CS events, persisted only as table rows; the in-loop
  representation is the per-model event idiom — what-code
  "approximately the index" (quasi-orthogonal: pairwise incomparable
  under dominance, mereologically inert) with **zeroed
  `.where`/`.when`** (MM_20M: `4+2+2`, total 8, matching its STM=8;
  widths are config, the zero-band law is architectural).
- `bind()`: append-only; **only the gate creates rows** (Stage 7's
  search-then-mint licenses it); gauge-oriented object row per Stage 5;
  evaluate-before-cache (`𝟘`-valued codes cached as
  definable-but-empty, queryable; ⊤-saturation detectable — spec §6
  hazard).
- Tests (§10.8, §10.9): full-rows-only enforced; append-only; deref
  indexed / ref unindexed (API audit: no object→name index anywhere);
  symbol codes quasi-orthogonal + zero-banded (property test:
  pairwise dominance-incomparable); `A ⊑ σ(A,B)` / `π(A,B) ⊑ A` at
  mint (random admissible weights); ⊥ caching; ⊤-saturation detection;
  search returns binding rows whose objects dominate the probe.

### Stage 7 — SymbolicSpace wiring: search-then-mint (spec §6, §7)

Files: `bin/Spaces.py` (`SymbolicSpace` `:12661`), new
`test/test_search_then_mint.py`.

- The reference half becomes SS's symbol organ; the interpret-as-word gate
  is implemented as **search, then mint on licensed miss** — a loggable
  decision; naming requires demonstrated reuse, never first sight.
  `adopt_stage0_evidence` (`:15724`) stays as ground-half memory,
  explicitly not naming.
- Cordon-as-discipline checks are *tests, not runtime guards* (spec §7):
  folds never create rows; semantic use of a non-row fails naturally;
  semantic π routes decode → fold downstairs → re-mint.
- **Serial-mode duals (spec §6 rev 2026-06-10c, "the table in time" —
  supersedes the earlier LHS/RHS languaging).** The grammar engages on
  both sides as duals: **SS side = analysis = binary split + shift**
  (the serial form of π — top-down part-making, constituents identified
  by codebook match); **PS side = synthesis = binary reduce + shift**
  (the serial form of σ, via the existing driver). When SS analysis
  reaches a code that is a **word in the SS codebook**, its object
  reference is looked up (deref through the paired `(orth, semantic)`
  lexicon row) and the referent is **placed on the PS side of the
  callosum — the shift step of a traditional parser**; the word is part
  of the sentence; the referent is not. Concretely: WordSubSpace's
  `_idea_buffer` (`ShortTermMemory`, `Layers.py:9770`, capacity 8,
  shift/reduce driver already inlined) is the **PS-side** idea
  workspace — **structurally unchanged** (already semantic-embedding
  content); add the **SS-side constituent stack** (8 slots of symbolic
  codes under analysis; word codes at its leaves, from lexer/codebook
  matches). The shift rule branches: content word → deref-shift;
  closed-class word → router/operator-marker bind (nothing shifts);
  unknown word → the search-then-mint gate fires *here* (unlicensed ⇒
  `a = 0` placeholder). Mention/quotation = stop-tagged shift of the
  word code itself, no deref. **Direction swaps the move sets** (spec
  §6 rev c): forward (comprehension) SS split|shift / PS reduce|shift;
  reverse (generation) PS split|shift / SS reduce|shift — the reverse
  shift is a **ref-search** (object → name; tip-of-the-tongue = failed
  reverse shift; paraphrase = split-further-until-nameable), placing
  word codes on the SS side for reduction into the sentence.
  **Two syntaxes, softly dispatched** (spec §6 rev c): an analytical
  syntax (split-side inventory) and a synthetic syntax (reduce-side),
  attached to roles not spaces (the direction swap moves them); both
  dispatch as rule-probability-weighted superpositions over candidate
  operators and graded matches, in the signal; the three-way shift
  branch is one soft computation's readout (unknown word = the
  `a → 0` end of the same lookup); discreteness lives only at the
  per-tick write and at mint (the bracketing commits at mint). In serial mode the `bind_streams` per-tick `2N` fusion is
  bypassed — the glue's fusion duty migrates into shift. Builds directly on
  `doc/plans/2026-05-29-stm-serial-parallel-modes.md` (mode knob,
  per-word predict-then-perceive loop, SS-analysis/CS-execution router
  split).
- Tests (§10.8 remainder + §10.11, `test/test_serial_stm_split.py`): gate
  logging; no row creation from fold paths; deref/decode round-trip
  preserves extent to tolerance; single-writer mutex per serial tick
  (one move: split | shift | reduce); shifted content is the semantic
  referent, never the word code (except stop-tagged mention-shifts,
  which preserve it); shifts fire exactly where SS analysis bottoms out
  at codebook words; marker words bind the router and shift nothing;
  reduce trace recoverable while its events live (no stored
  derivational identity — spec §8 resolved: identity is content +
  participation); serial reduce-chain extent ≈
  parallel σ extent on associative content; serial split-chain ≈
  parallel π analysis on the dual side; parallel mode leaves both
  serial buffers untouched.

### Stage 8 — Two towers + registration as drift-keeper (spec §5)

Files: `bin/Spaces.py`, `bin/References.py`, new
`test/test_registration_drift.py`.

- **Organ mapping resolved (rev 2026-06-11): the towers ARE the two
  codebooks.** Extent codebook ≡ the PS codebook; intent codebook ≡
  the SS codebook — the same two used for incoming IS recognition; no
  new `Basis` objects, no repurposed `analysis_store`. All synthesis
  methods share PS's, all analysis methods share SS's (every space
  carries such a codebook by this stage). Consequence to wire and
  test: **recognition is tower lookup** — IS matching against the PS
  codebook is matching against cached extents. Symbol-wholes and
  symbol codes carry the zero-band signature (spec §6/§7).
- `σπσ = σ` registration loss beside CWCE, now with its second duty:
  **maintenance** — after perturbation/training steps, minted cells must
  re-register (dominance over parts restored) or be flagged. This is
  also the mechanism of **isomorphic pressure** (spec §5): asserted
  pairs from corpus predications are registration constraints.
- Tests (§10.5): loss decreases on a toy corpus; non-degenerate fixed
  points; drift case — perturb, train, verify re-registration;
  predication-pressure toy — record taxonomic truths, train, verify
  geometric dominance moves toward the asserted order; asserted
  pairs verified by dominance.

### Stage 9 — Cutover + reconciliation

- Flip `<meronomy>` to `on`: PS `self.sigma` → `SigmaLayer2`
  (`Spaces.py:8406–8438`), SS `self.pi` → `PiLayer2`
  (`:12911–13091`) — meronymic slots only. Non-meronymic consumers of the
  legacy layers (Lift/Lower C-tier rule layers, grammar ops) keep the odds
  path until separately decided.
- Re-baseline the suite; re-pin XOR fixtures if numerics shift (expected;
  same seed-1 + CPU discipline).
- Land the doc reconciliations (§2 below).
- Gate: zero failures; spec §10.1–§10.11 each covered by a named test from
  Stages 0–8 (traceability table in the PR description).

## 2. Doc reconciliations (spec wins)

| Doc | Conflicting claim | Action |
|---|---|---|
| `doc/Mereology.md:42–66` | clipped-cosine `part()` as *the* parthood relation | Reframe: graded retrieval surrogate; exact dominance (`Ops.partOf`, `Layers.py:11713`) is ground truth; `part()` total and form-level even across the reference boundary (spec §1, §7). Add asserted-vs-incidental split and its cordon duty. |
| `doc/Mereology.md:109–121` | fusion = elementwise max (LUB) | Keep as the order-theoretic join; the learned σ-fold satisfies `σ ≽ max` (§10.1) — join vs parametric whole-maker. |
| `doc/Mereology.md:77–97` | region partition incl. exact `equal == 0` underlap | Note: no abstraction classifier consumes this (withdrawn, spec §9); the partition remains a scoring vocabulary only. Contiguity is sequential (`Mereology.Contiguous`), never dimensional (spec §2). |
| `doc/Logic.md:24–26, 270–273` | negation = `−x` plus `non()` withdrawal | Ground in spec §3: sign flip licensed only at the reference tier (sign-vacancy argument); `non()`'s withdrawal reading = prasajya magnitude-lowering, gated; RadMin/RadMax remain signed-scalar kernels at their tier. |
| `doc/Logic.md` (odds, via Language.md) | odds/multiplicative fold embedding | Retired from the meronymic path; lives on only in legacy `PiLayer` for non-meronymic consumers (spec §4). |
| `doc/Logic.md:280–324` (§Luminosity) | `luminosity = \|\|relu(min(truths))\|\|` as *the* truth-set coherence measure | Split the two measures (spec §3 rev 2026-06-10b): **truth-set luminosity** (`Mereology.Luminosity(truth_layer=…)` → `TruthLayer.luminosity`) is the catuṣkoṭi coverage measure over the codes — per conceptual dim `(T_k, F_k)` pole coverage, `mean_k[(T_k−F_k) − min(T_k,F_k)]` in `[−1,1]`, order-independent, **no decode pullback**. The meet/GLB form survives only as the orthogonalization criterion (`_luminosity_without`) and `darkness`'s mirror. Fusion-as-LUB subsection stays as-is; the repeated-vs-leading bivector layout caveat stays. Update the diagram (`diagrams/luminosity.svg`) or its caption. |
| `doc/Reasoning.md:63, 90` | extrapolation gated on luminosity non-decrease; additive TruthLoss vs multiplicative luminosity | Roles unchanged; note the gated quantity is now the §3 rev-b catuṣkoṭi measure (signed area − conflict), not the meet norm. |
| `doc/Spaces.md` | PS-σ / SS-π (already correct, 2026-06-09) | Add: gauge-embedding × scalar-certainty encoding (§3), nameless factored injection (§3), unified index space with ground/reference halves (§6), `PiLayer2`/`SigmaLayer2` slot binding. |
| `doc/plans/BivectorRetirementPlan.md:24–29` | K3: signed scalar as the only inter-component carrier | Extend, not contradict: pole-quotient lookup licensed at the reference half only. (Rev 2026-06-10b withdraws the earlier "+ one per-symbol κ inside TruthLayer" extension recorded here: contested-ness is **measured, not stored** — the conflict region `min(T,F)` of the §3 truth measure.) |
| `bin/embed.py` docstrings | `_wrap_unit_ball` / `_pole_aligned_score` scope notes | Update: wrap licensed for embedding position (no truth in stored coordinates); pole-quotient = the reference-half lookup law (spec §3). |
| `bin/etc/SigmaPi.py` | archived XOR experiment | No action (already in `etc/`). |

## 3. Risks

- **Drift vs registration (primary).** Geometric encoding of parent/child
  is exact at mint, maintained only by the registration loss; Stage 8's
  drift test is the canary. If maintenance proves weak, fall back to
  re-snap-on-read for hot cells (cheap, local), not to stored order edges.
- **Search cost of `ref`.** Recall-as-search is the designed asymmetry
  (tip-of-the-tongue is a feature), but probes must be batched/bounded;
  the recommender path is already matmul-shaped.
- **Table growth.** The half-table census assumes selective minting; the
  gate policy (reuse-justified naming) is the enforcement point — log and
  monitor mint rate from day one.
- **Numerics at the corners.** `d ≥ 1` amplifies negative log-mass;
  `EPS_LOG` floor + `D_MAX_STABLE` clamp bound it; §10.2 pins the corners.
- **Cutover numerics.** XOR fixture re-pinning expected at Stage 9; same
  determinism discipline (seed-1 + CPU).
- **Mutex granularity.** A whole-substrate serial mutex parks perception
  while parsing. The decided default scopes the mutex to the STM
  single-writer rule (spec §6: one move per tick, split | shift |
  reduce); preattentive parallel synthesis beneath serial parsing, with
  the §3 truth-measure conflict region (taint) as the preemption
  signal, is deferred to the grammar-ops corrective pass. If the
  default proves too coarse, the mode flag moves from model-global to
  per-workspace.

## 4. Open items — author decisions (resolved 2026-06-10/11)

1. **Opaque-unit mark — RESOLVED (2026-06-11): neither.** No branch
   tags, no singleton cells, no order maintenance. Symbols are atomic
   table rows (the lasso *is* the binding); opacity is by construction
   (atoms have no insides); the table holds full word/object rows only
   (spec §6 rev 2026-06-11).
2. **Stage 8 organ mapping — RESOLVED (2026-06-11): the towers are the
   two codebooks.** Extent ≡ PS codebook, intent ≡ SS codebook, shared
   with incoming IS recognition; no new `Basis` objects, no repurposed
   `analysis_store`.
3. **`D_MAX_STABLE` default — CONFIRMED (2026-06-11): `4.0`**,
   per-model overridable via `<meronomy dMaxStable>`.
4. **Lift/Lower and grammar-op kernels — RESOLVED (2026-06-11) as a
   per-duty choice, deferred to the grammar-ops corrective pass.** The
   operators are NOT legacy — they are permanent C-tier architecture;
   only the kernel they wrap is in question. Either kernel is
   admissible: the membership kernel adds order preservation (tier
   ladder: order-0 prototypes → order-1 generalizations → order-2
   discontiguous sets of generalizations; dimension change is
   orthogonal to tier) and cannot cancel features (`W ≥ 0`); the odds
   kernel is the `p+n=1` sharpening law for evidence combination.
   Decide per operator duty at the corrective pass.
5. **Module name — default stands: `bin/References.py`** (apter than
   ever: rows are references — word→object bindings).

6. **Hyperintensionality / derivational identity — RESOLVED
   (2026-06-11): identity is content + participation.** No stored
   derivational identity: a multiply-derived code is content-queried
   (stabilizing bottom-up) and its participation in the conceptual
   space provides the meaning, not the encoding process that installed
   it. Lossless where it matters: meaning-relevant derivation
   differences surface as content differences (§4 laws); sense
   distinctions survive as distinct table rows; associative
   re-bracketings rightly collapse. Provenance/replay, if ever needed,
   is an episodic/event concern (spec §8).

## 5. Acceptance traceability (Stage 9 gate artifact, 2026-06-11)

Spec §10 criterion → named tests (all green at cutover):

| § | Criterion | Tests |
|---|---|---|
| 10.1 | Order theorems | `test_pi2_sigma2_folds.py::test_binary_pi_below_min` / `test_binary_sigma_above_max` / `test_unary_contraction_and_extension` / `test_monotone_transport` / `test_binary_zero_operand_corner_still_ordered` |
| 10.2 | Roots | `test_pi2_sigma2_folds.py::test_pi_absorber_is_a_theorem` / `test_sigma_absorber_is_a_theorem` / `test_pi_identity_at_init` / `test_sigma_identity_at_init` |
| 10.3 | Encoding & gauge | `test_reference_lookup_gauge.py` (all: gauge orient ×3, relabeling invariance, certainty survival, token unaffected, knob-gated sphere snap ×4) |
| 10.4 | De Morgan | `test_pi2_sigma2_folds.py::test_de_morgan_exact_with_shared_kernel` / `test_sigma_owns_no_weights`; corner round-trips: `test_pi_roundtrip_at_zero_corner_under_eps_log` / `test_sigma_roundtrip_at_zero_corner` |
| 10.5 | Registration & drift | `test_registration_drift.py` (loss decreases, non-degenerate fixed points, drift flagged + re-registered, predication pressure) |
| 10.6 | Interface law | `test_interface_factoring.py` (no stimulation ⇒ a=0; negative half unreachable; χ exact & once; row-span crossing; knob paths) |
| 10.7 | Truth measure (rev b) | `test_truth_luminosity_codes.py` (15: corners, conflict, coverage, trust, order-independence, no pullback, range, domain boundary) |
| 10.8 | Table discipline | `test_reference_table.py` (full rows, append-only, gate license, no reverse index, search-by-dominance, symbol codes) + `test_search_then_mint.py` (gate logging, placeholder-never-row, folds never create rows) |
| 10.9 | Two engines | `test_reference_table.py::test_minted_whole_is_searchable_by_its_parts` / `test_bottom_extent_cached_definable_but_empty` / `test_top_saturation_detected`; `test_registration_drift.py::test_drift_flagged_and_re_registered` |
| 10.10 | Weight law | `test_contractive_linear.py` (18: init, adversarial descent/ascent, regression, ergodic noise, round-trips, blocks) |
| 10.11 | Mode coincidence | `test_serial_stm_split.py` (constituent stack, split, shift semantics, mention, marker, single-writer, deref round-trip, reduce-chain ≈ parallel σ, parallel leaves stacks dark) + `test_meronomy_laws.py` (chart + guard) |

Stage gates: every stage landed with the full suite at zero failures,
run from `basicmodel/`; the Stage 9 cutover (this rev) flipped
`<meronomy>` to `on` in `data/model.xml` (+ `data/model.xsd` element)
and re-ran the full suite green.

## 6. Next step after this plan (recorded, out of scope)

**Grammar-ops corrective pass.** `lift()` / `lower()`
(`Language.py:2286` / `:2490`) are total today — they apply over any
operand pair; "colorless green ideas sleep furiously" is the canonical
composition that a declared **validity mask** would flag (MeronomySpec §4).
Requirement notes are placed in both class docstrings. Mechanism seeds:
the per-call `gate` low-rank operator slicing (`PiLayer.forward`,
`Layers.py:3491` → `_d_effective`, `:1818` — already used by the rule
layers) and GrammarLayer's `tier` / `rule_probability` gating
(`Layers.py:1933`), which gates which rules fire where but not which
operands a rule validly takes. Author corrections (2026-06-11) scoping
the pass:

1. **Syntactic validity is the grammar files' job, not the masks'.**
   Categories live in `.grammar`; the standing migration is all
   grammars to the role-collapsed format (roles = operator argument
   position; `data/role_collapsed.grammar`, plan
   2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar).
   The **lift/lower masks** are LEXICAL modulation of shared operator
   matrices — e.g. walking vs running on ONE verb matrix, the per-verb
   mask selecting its slice (seed: the low-rank `gate` slicing on
   `_d_effective`; the lexical item's code is the mask source). This
   constrains item 2: the kernel choice must preserve the gate-slicing
   surface (both kernels and the MeronymicFoldAdapter do).
2. **Per-duty kernel audit** for the non-meronymic operators (plan §4
   item 4): order-preserving duties → membership kernel; evidence
   combination/sharpening → odds kernel; tier-crossing duties → the
   adapter pattern.
3. **The syntaxes do NOT unify through De Morgan.** De Morgan pairs
   AND↔OR (π↔σ — the Boolean mirror at the vector level, one kernel
   per pair, as implemented in Stage 2) WITHIN each syntax. The
   analytical/synthetic relation is the DIRECTION axis: analysis = π
   forward + σ reverse; synthesis = σ forward + π reverse. Inventory
   sharing across syntaxes, if adopted, is through INVERTIBILITY of
   rules; De Morgan economizes kernels, invertibility economizes
   rules — orthogonal sharings.
4. **Semantics enters as attention over symbols, not semantic rules.**
   The grammars stay syntactic; codebook priming (attention modes
   `off|primer|second-order|low-rank`; primed-reverse-generation boost
   weights) supplies meaning-sensitivity. Work items: mirror the
   SS-side priming into BOTH codebook towers (towers-as-codebooks
   makes priming = boost weights over rows; extents deserve it as much
   as intents), and combine both grammars with a SINGLE INTENT — one
   current-intent code priming both towers, weighting both syntaxes'
   superpositions toward context-relevant symbols.
5. **Mutex granularity**: preattentive parallel synthesis beneath
   serial parsing with the §3 truth-measure conflict region as the
   preemption signal; preemption policy (threshold/hysteresis;
   abort / checkpoint / frame-split) decided in the pass; per-workspace
   mode flag as the fallback.
