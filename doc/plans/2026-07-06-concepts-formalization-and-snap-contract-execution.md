# Concepts / Snap Contract — Execution Plan (T0–T6)

> **STATUS: EXECUTED 2026-07-06 — T0–T6 all landed and full-suite gated
> (3008/0 → 3031/0 + T6 gate), same session as authorship.** Deviations
> from plan, resolved with Alec in-session: T0 found 11 stale-`join`
> configs (not 6); T1d became a structural split (concept store moved OFF
> SparseLayer onto the renamed ConceptualAttentionLayer — Alec's call);
> T3's peel is ORTHOGONAL MP (lstsq refit — plain projection-MP is inexact
> on non-orthogonal rows) and `cs_snap_order0`'s nonneg was deliberately
> KEPT (presence seed is mereological; sign lives on the weights — Alec's
> factor-concepts-vs-wholes rule); T5's "no shadow" scare was a probe
> artifact (identity-init LDU + ergodic set_sigma — Alec caught it); the
> §2.3 bar is demonstrated with trained folds. T6's typed definition keeps
> BOTH frontiers (head = tightest cover = the coverer covering no other
> coverer; modifiers = differentia, never implied away by the head).
> Grammar-surface compression (Method-2/Task-4 meter) remains with the
> serial-derivation plan. All work uncommitted — Alec commits.

> **ORIGINAL STATUS: EXECUTION PLAN 2026-07-06 — self-contained; written for a fresh
> implementation session with no conversation context.** Companion design:
> [2026-07-06-concepts-formalization-and-snap-contract-design.md](2026-07-06-concepts-formalization-and-snap-contract-design.md)
> (read §1 framework, §2 snap contract, §3 fidelity bars first). This doc
> is the design's §4 expanded task-by-task, with every file:line anchor
> **re-verified on `main` @ 3a982be (2026-07-06)**, the drifted anchors
> corrected, the breakage lists enumerated, and the minor open calls pinned
> so nothing needs a design decision mid-task. Execute T0 → T6 straight
> down. Design calls already resolved by Alec (§5 of the design): sign
> carrier = the symbol scalar; sparsity = rank-ordered soft-then-hard L0;
> storage domain = small-magnitude init. Only λ (empirical, T4) and the
> read-time ε (pinned to a default below, revisable) remained — both are
> handled in-plan.

## 0. Ground rules (read before touching anything)

**Environment / gates.** Targeted per-task gate:

```
cd basicmodel && PYTHONPATH=test:bin BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager \
  .venv/bin/python -m pytest <files> -x -q
```

Add `RUN_SLOW=1` when the gate includes a slow test — in particular
**`test_mm20m_grammar_derivation_roundtrip` (test/test_reconstruction_roundtrip.py:577)
only runs under `RUN_SLOW=1`** (~11s alone; the full file ~2:22 with
RUN_SLOW). `make test` (serial, canonical) only at a task's final gate on
a quiet tree (~15 min). **Never edit source while a suite is running**
(inspect.getsource-based tests fail spuriously). Inf/NaN fail loud;
value-pins updated honestly.

**Expected suite noise (not failures):** `test_mm20m_xor_blind_roundtrip`
(test/test_blind_decode.py:206) is `xfail(strict=False)` + RUN_SLOW-gated —
an Alec-deferred .where-band knob call; it reports skip (default) or xfail
(RUN_SLOW). If it ever XPASSes that's a *signal* (band matured), not a
failure. The baseline also carries 53 skipped / 32 xfailed / 7 xpassed.

**Commit policy.** Alec makes all commits himself. Complete each task (or
named sub-task) to a verified stopping point on the working tree, then
**STOP and hand off for the commit before starting the next task**. The
stop points are: after T0, T1a, T1b, T2, T3, T4, T5, T6. This is what the
design's "its own commit + full-suite gate" means operationally.

**Anchor policy.** Every file:line in this doc was verified on 3a982be.
T1 renames ~250 lines across bin/ (incl. 9 sites in Models.py), so **from
T2 onward treat line numbers as advisory and navigate by symbol name**
(each anchor below gives both). Do NOT copy anchors from todo.md — at
least five are stale (property_basis 2599→2762, materialize 6718→6907,
out_slot 7848→7859, center_k 3020→3015, recon_bench.py:206 → the real
`where_recovery_rate` now lives at recon_bench.py:69), and todo.md's
hyperlinks point at the iCloud copy, not the git tree.

**Op-naming translation table (CRITICAL TRAP).** The companion doc
[2026-07-04-union-difference-concept-ops.md](2026-07-04-union-difference-concept-ops.md)
predates the 2026-07-05 naming pass. Current reality:

| old name (stale doc) | current name | where | character |
|---|---|---|---|
| union (additive) | **chunk** | `ChunkLayer`, bin/Language.py:2264 | additive, residual-bearing, `<PartSpace>` structural op |
| difference (rule) | *retired as a rule* | static helper `ChunkLayer.difference`, bin/Language.py:2323 | exact residual |
| join | **union** | `UnionLayer`, bin/Language.py:2135 | LOSSY lattice max (RadMax), `invertible=False` |
| — | sum / product | `SumLayer` :2383 / `ProductLayer` :2426 | CS arithmetic pair |

`test/test_union_difference_ops.py` kept its filename; its
`test_registry_and_fixity` asserts `join`/`difference` are GONE from the
registry. An implementer who greps for `UnionLayer.peel` finds the lossy
lattice op — the exact wrong one. The peel is **`ChunkLayer.peel`**.

**Live-vs-dark correction (design doc caveat).** The meronomy abs-snap
branch (`_snap_content`, gate `meronomy_enabled() and not self.monotonic`
at bin/Spaces.py:2038) is **effectively LIVE under every shipped config**:
data/model.xml:71 sets `<meronomy dMaxStable="4.0">on</meronomy>` and
model.xml is the defaults file merged under every per-model XML
(model.xml:76 also sets `<monotonic>false</monotonic>`). The in-code
"dark until cutover" comment at Spaces.py:2045 is stale. Consequence: T3
edits live-path code; the S2 round-trip is the mandatory regression after
every T3 edit. (Do not confuse the gate with the `<synthesis>`/`<analysis>`
*meronomy* enum values in MM_20M_grammar.xml etc. — different knob.)

## T0 — Baseline repair (the Gate-4 close)

**Fresh `make test` on 3a982be (2026-07-06): 3 failed / 3005 passed /
53 skipped / 32 xfailed / 7 xpassed, 915s.** The tree is NOT green, so
T1's "byte-identical" verification is uninterpretable until this is fixed.

**Diagnosis (verified).** All 3 failures share one root cause: the
2026-07-05 `join`→`union` naming pass updated the `data/*.grammar` files
(e.g. data/xor.grammar:42 now `union.forward`) but **missed the inline
`<grammar>` blocks in six data/*.xml configs**, which still declare the
retired `join`:

- data/MM_xor_fixture.xml:81 `<S>S = join(S, S)</S>` ← the one the 3 red
  tests load
- data/MM_bpe.xml:75, data/MM_5M_IR.xml:97, data/MM_boolean.xml:128 (+
  prose at :96/:106), data/MM_5M_AR.xml:91, data/MM_shamatha.xml:51

The grammar parses the rule (method_name=`join`) but the registry has no
`join` layer, so `insert_operations` / build skip it → 
`test_operator_superposition_resolves_against_codebook`,
`test_operations_inserted_into_ws_codebook`, and
`test_build_auto_inserts_operations` all assert
`{'intersection','join','not'}` coverage and get `{'intersection','not'}`.

**Fix.** Rename `join(` → `union(` in the six inline grammar blocks (and
the MM_boolean.xml prose comments). Semantics-preserving: `union` is the
same RadMax layer `join` used to name. Note the rule is currently
*inert* in those configs (no layer resolves), so the fix makes it live
again — the intended semantics restored, but check pins.

**Verify.** The 3 red tests green; then `make test` — MM_xor_fixture is
loaded by ~22 test files and the other five configs by ~9 more, so a
full-suite run is the honest gate. If a value-pin shifts in a test using
those configs, evaluate and update honestly (expected: none or tiny —
the op becomes insertable, it doesn't change trained paths in fixtures).
**Expected post-T0 baseline: 3008 passed / 0 failed.** Record the count —
it is the reference for every later gate. **STOP for commit.**

## T1 — Terminology rename pass (no design call; names not behavior)

Three sub-parts; T1a and T1b are separate stop-points, each full-suite
gated. Byte-identical behavior is the bar. **Exclude `doc/old/` from both
the rename AND the post-rename leftover greps** (frozen archives — 3 files
mention insert_symbol, ~8 mention nObj; leave them).

### T1a — `insert_symbol` → `insert_whole`

The design says "~100 source sites — disambiguate each definition first."
**Census result: the disambiguation is already done and the answer is
global.** There is EXACTLY ONE definition —
`WholeSpace.insert_symbol` (bin/Spaces.py:17510, class at :16702) — and it
inserts a WHOLE (a fresh `WS.subspace.what` codebook row, `_pos_kind`
"ws"). There is NO genuine SS symbol inserter named insert_symbol
anywhere (SymbolSpace at bin/Language.py:12482 defines no insert_*). All
100 occurrences (bin/ 28: Spaces.py 27 + Layers.py 1; test/ 72) resolve to
the WS whole-inserter. No dynamic string refs, no getattr dispatch, no
test asserts on its exception text, method names never enter state_dict —
**a word-bounded global rename is safe.** `insert_whole` collides with
nothing.

Include in the same pass:
- The runtime error-message strings inside the method
  ("WholeSpace.insert_symbol requires…" at Spaces.py:17525/17542/17560) —
  rename the text so diagnostics stay accurate.
- Inverted test naming that would read wrong post-rename:
  test/test_mereology_raise.py:47/77/97 bind `ss = _whole_space()` (an
  `ss` that is a WholeSpace — rename the variable, e.g. `ws`);
  test/test_two_codebook_meta_taxonomy.py:126 docstring calls a WS row an
  "SS row" (fix the prose).
- doc/SymbolFirewall.md:54 cites insert_symbol with stale anchors
  (says Spaces.py:14289; def is 17510; insert_meta cited 14730, actual
  ~18017) — update name + anchors.

**Verify:** full suite == post-T0 count. **STOP for commit.**

### T1b — `nObj` → `nIdeas` (+ ObjectSubSpace → IdeaSubSpace)

**The design's "~159 sites" is a substring count that INCLUDES the
distinct, live identifier `nObjects` (~20 lines) — a blind replace
produces `nIdeasects` and breaks the config layer.** `nObjects` is
load-bearing (bin/util.py:1051 `def nObjects` property; bin/Models.py:568/
575 config writeback; comments at Spaces.py:743/897/930/1046/7407/7560/
12282; model.xsd:64 comment) and must survive untouched.

- Use word-boundary **`\bnObj\b`**: true scope = 139 lines / 149
  occurrences (bin 120: Spaces.py 109, Layers.py 10 — wait, per-file split
  is Spaces.py 109 + Layers.py 10 + Models.py 9 + data.py 1 + util.py 1 =
  130 bin lines; test 19 lines / 29 occurrences).
- **Checkpoint/config-safe (verified):** nObj is NOT an XML tag (zero
  hits in data/*.xml; model.xsd has only the `nObjects` comment), NOT a
  state_dict key, parameter, buffer, or quoted string — purely local
  variables (dominant pattern `nObj = self.outputShape[0]`) and function
  parameters (Spaces.py:1457 encode_tokens, :1516 tokens_to_decoded,
  :10887/:11020 `_bpe_emit*`, Layers.py:11085 segment_words).
- `ObjectSubSpace` (bin/Language.py:8702, refs: bin 14 / test 14 / doc 38)
  → **`IdeaSubSpace`** (PINNED-BY-PLAN; the design says the "object=slot"
  naming follows the idea-slot rename and names no target — `nIdeas`
  holds per-word **ideas**, so the subspace of those slots is
  IdeaSubSpace). No collision (zero grep hits). Sweep the doc/ references
  outside doc/old/ in the same pass.
- Live doc prose in the same pass: doc/Spaces.md:172 ("the [B, nObj,
  nDim]…"), doc/plans/2026-06-23-reasoning-live-wiring.md:33.

**Verify:** full suite == post-T0 count (byte-identical pure rename).
**STOP for commit.**

### T1c — ladder-terms prose sweep (fold into T1a or T1b commit)

- doc/percept-hypercube.md — 6 "concepts/symbols" pairing lines:
  153, 220, 272, 308, 310, 329 → the design §0 ladder terms (concepts =
  CS regions; symbols = SS signed presence scalars; the pairing usually
  means "the signed sphere stores" — reword per site, don't mass-replace).
- todo.md:56 "symbol/truth prototypes" (the WS codebook description in
  the property_basis item) → whole/truth prototypes wording. (The
  property_basis *mechanism* itself is untouched — separate deliberate
  step per that todo.)

### T1d — Split concept structure out of `SparseLayer`; rename the concept layer to `ConceptualAttentionLayer` (Alec, 2026-07-06)

**Structural, NOT a pure rename** (its own stop-point; run last in T1 on a
quiet tree). Behavior-preserving (same methods, same objects, relocated),
so the bar is full-suite green at the post-T0 count — not a byte-identical
diff.

**Why.** The intended SparseLayer→AttentionLayer rename was done by a
prior session as a SUBCLASS (`class AttentionLayer(SparseLayer)`,
bin/Layers.py:4708) — so the base name survived AND the base is not clean:
`SparseLayer` (bin/Layers.py:4405) bundles a generic COO substrate with a
large concept-specific block. Verified: the concept relation-store API is
called from **17 sites, all in bin/Spaces.py**; bare `SparseLayer(...)` is
built only at the `ConceptAllocator.layer` fallback (:4768, itself concept
code) and in the generic unit test. Alec's resolution (2026-07-06):
**move the concept-specific block onto the renamed concept layer; keep
`SparseLayer` a genuinely generic substrate.**

**End state.**

- **`SparseLayer` (base, keeps the name)** = generic COO sparse-linear
  autoencoder substrate ONLY: `values`, `_rows/_cols/_init_vals/_index`,
  `roles`/`role_slice`, `add_edge`/`remove_edges`, `_grow_values`,
  `_indices`, `_matmul`/`_apply_shape`, `forward_linear`/`forward`/
  `reverse`, `nnz`/`_device`, kernels, `forbid_self_edges`. Genericize the
  concept prose: the docstring ("percept/symbol → concept map of the
  ramsified conceptual space") and the `add_edge` "Quine atom" error
  message (:4508-4510) lose concept vocabulary (keep forbid-self-edges as
  a plain graph property). This is exactly the surface
  test/test_sparse_layer.py already tests generically.
- **`ConceptualAttentionLayer(SparseLayer)`** = today's `AttentionLayer`
  (bin/Layers.py:4708) renamed, ABSORBING the concept block moved up from
  the base:
  - state: `_constituents`, `_tensor_rows`, `_row_next` (:4451-4453);
  - the "discrete relation store" methods (:4597-4694): `ensure_row_key`,
    `pop_row_key`, `put_row_key`, `constituents`, `add_constituent`,
    `remove_constituent`, `set_role_constituents`, `clear_constituents`,
    `count_role`, `row_is_identity`, `embed_pair`, `discretize_row`,
    `assign_row`, `row_of`;
  - `hebbian_strengthen_row` (:4696-4705);
  - keeps `wave_step` + the square [N×N+1] ctor it already has.

**The one obstacle — `ConceptAllocator.layer()` (bin/Layers.py:4755-4771)
must return a `ConceptualAttentionLayer` in BOTH branches.** Today the
square branch builds `AttentionLayer(n_out)` (:4766) and the fallback
builds a bare `SparseLayer(n_in, n_out, roles=…)` (:4768); the allocator
then calls the record API on whatever it returns. After the move, the
bare-SparseLayer fallback would have no record API. Fix: the fallback also
builds a `ConceptualAttentionLayer`. Safe because the record-store methods
are shape-independent host-side dicts, and the degenerate/roles fallback
is the NOT-sparse-active path that never calls `wave_step` (verified: the
only wave_step caller is `cs_forward_content`, Spaces.py:14515, on the
square store). So give `ConceptualAttentionLayer` the flexible base
`__init__` signature (nInput, nOutput, …, roles, forbid_self_edges) plus
`wave_step` and a square convenience constructor; rewrite :4766 to the
square form and :4768 to `ConceptualAttentionLayer(n_in, n_out,
roles=roles)`.

**Name-collision note:** plain `AttentionLayer` and `QKVAttentionLayer`
(:5307) both exist — that's why the concept layer takes the qualified
`ConceptualAttentionLayer`. Zero grep hits for the new name today.

**Touch sites:** class defs + the moved block (Layers.py); the two
`ConceptAllocator.layer` branches; comments Models.py:3466 / Spaces.py:
14001; test/test_sparse_layer.py — SPLIT it: generic COO tests stay on
`SparseLayer`, add/move the record-store + wave tests onto
`ConceptualAttentionLayer` (rename the file, e.g.
test_conceptual_attention_layer.py, or split into two). doc/Architecture.md
(154/193/357/387) + doc/Spaces.md prose that frames SparseLayer as "how it
works" for concepts — reword to the split. The 17 Spaces.py record-API
call sites are UNCHANGED (they call through `store_of(cid)` /
`self.layer()`, which now returns a `ConceptualAttentionLayer`).

**T4 note:** T4's penalty targets the concept wave layer's `.values`
Parameter — after T1d that is `ConceptualAttentionLayer.values` (inherited
from `SparseLayer`); T6's `row_is_identity`/`_constituents` are now on
`ConceptualAttentionLayer`. Re-anchor by name.

**Verify:** full suite == post-T0 count; the split test files green.
**STOP for commit.**

## T2 — WS `<initScale>` + measurement (fidelity leg §3.2)

Goal: WS rows seed at ~0.02 magnitude so depth-8 σ/π folds stay in the
linear regime (tanh saturation math in design §3.2); PS radix is already
there (`RadixLayer.insert` normal_(0, 0.02), bin/Layers.py:10483-10486 —
verified; the S2 round-trip green at that scale is the viability proof).

**Init sites (all verified; T1a will have renamed insert_symbol →
insert_whole — navigate by symbol):**

1. `WholeSpace.insert_whole` None-seed: `uniform_(-1.0, 1.0)` at
   Spaces.py:17545-17548, written at :17563-17564.
2. `Codebook.addVectors` prefill — **TWO branches, and the WS symbol
   codebook takes the customVQ one** (the design's ":3441" is only the
   generic branch): (a) customVQ at Spaces.py:3411-3431 — dot-product
   mode `F.normalize` unit-L2 (:3420), else per-row max-abs rescale into
   [-1,1] (:3422-3423) + clamp (:3430); (b) generic at :3441-3445 —
   per-row `self.normalize` unit-norm. Both land rows at order-1
   magnitude. Patching only (b) leaves the WS symbol codebook — the whole
   point of T2 — untouched.
3. The creation call sites that know their config section:
   `WholeSpace._build_what_basis` create (Spaces.py:19011-19022,
   customVQ=True, monotonic=True at :19018) and the `analysis_store`
   create in `WholeSpace.__init__` (Spaces.py:16879-16894, under the
   pinned RNG seed 0x57A6E0 — preserve the seeded determinism).

**Threading mechanism (Codebook is section-agnostic — it cannot read
config itself).** Pass `initScale` as a `create()` kwarg (or set an
attribute on the basis before `addVectors` runs), supplied by the caller
that knows `config_section` (`_build_what_basis` / `WholeSpace.__init__`).
Default = current behavior for every codebook not explicitly given a
scale — `Codebook.create` is shared by ALL codebooks (CS, OutputSpace,
category VQ); a global change leaks everywhere. Config plumbing follows
the lrScale template: XSD element in **wholeSpaceType** (data/model.xsd:
990; lrScale precedent at :1137 — every config validates against
model.xsd via util.py:1314/1330), read at point of use via
`TheXMLConfig.space("WholeSpace", "initScale", default=None)` (generic
accessor util.py:1106-1163; construction-time read precedent: the LBG
knobs at Spaces.py:16837-16851).

**Mode invariants:** in `use_dot_product` mode the unit-norm prefill IS
the selection contract (argmax dot == cosine) — the knob must either skip
that mode or renormalize after scaling (scale is then selection-inert but
domain-honest); the monotonic path clamps [0,1] then L2-normalizes
(`Basis.normalize` :2086-2087) — keep the [0,1] percept-store semantics
(Codebook.getW UNORM-STE) coherent.

**Data-scale writers (the attractors that will silently undo a small
init — the design's "pair with checking those writers", enumerated):**

| writer | site | T2 action |
|---|---|---|
| `insert_whole(init_vec=…)` raw copy | Spaces.py:17563-17564; callers :13691 (_maybe_autobind_meta), :13844 (_autobind_word_wholes), :15310, :18171 (insert_meta seed), :18408 (maybe_raise_order), :18627 (maybe_split_lbg) | **guard:** when initScale is set, rescale the incoming seed to initScale·unit (direction kept) |
| `insert_meta` idempotent-revisit EMA blend — **the strongest attractor** (fires on every re-presentation) | Spaces.py:18094-18107 `blended = (1-ema_f)*old + ema_f*fv` | **measure-only** in T2 (it IS the learning signal); watched by the radial-spread probe |
| `adopt_symbolic_evidence` / `adopt_stage0_evidence` raw evidence into virgin analytic-store rows | Spaces.py:20396-20398 / :20325-20327 | measure-only (analytic store, not the space-owned W) |
| recon-gradient training of the promoted vq.codebook | Models.py:5998-6013 | measure-only (that's training) |

(The quantize-EMA refresh is NOT live for WS: `ema_update=False` hardwired
for every WholeSpace stage at Models.py:5988-5997; gate read at
Spaces.py:3478-3479 and Layers.py:14877-14878 — verified. So a small init
STICKS on WS absent the writers above.)

**Verify.** The design's cited vehicle is misleading:
`test/test_union_difference_ops.py` is **pure-tensor** (ChunkLayer/
SumLayer statics + a _BasisShim) — it never touches space init and passes
regardless of initScale; its "depth-8" is a `max_parts=8` peel budget,
not composition depth. Keep it as the untouched-contrast regression.
**NEW test work:** (a) a depth-8 composition-chain residual probe wired to
real space init (build a small config with initScale=0.02, insert rows,
fold to depth stmCapacity=8, assert the residual/chunk survives where the
1.0-scale init saturates); (b) the radial-spread probe from design §3
(composites separate by order instead of saturating; log the row-norm
distribution so the writer attractors are visible). Full suite green
(knob default-off ⇒ byte-identical elsewhere). **STOP for commit.**

## T3 — Signed peel (un-discard + coefficient)

### The FINAL peel contract — pinned once, so T3/T4/T5 don't churn the
### same function three times

```
ChunkLayer.peel(whole, basis, max_parts=8, eps=PEEL_SUPPORT_EPS,
                prototypes=None)
  -> (parts: list[(row: int, coeff: float)], residual: Tensor)
```

- `max_parts`: existing loop bound (Language.py:2351/2362). Production
  callers thread `model.conceptualSpace.stm_capacity` into it (T4 —
  published attribute at Spaces.py:12826; **there is NO existing
  stmCapacity↔peel coupling to reuse; this is new wiring**).
- `eps`: NEW support/stop threshold. **Termination contract change:**
  with the sign-break removed, the only stops left would be the
  residual-norm floor (:2363) and max_parts — every peel would run to
  max_parts. New stop: `|sims[best]| <= eps` breaks.
  **PINNED-BY-PLAN: `PEEL_SUPPORT_EPS = 1e-2`**, a named module constant
  (config-overridable later per design §5 open-1; the dead-zone ε is
  explicitly a minor read-time knob, decoupled from sparsity).
- `prototypes`: optional pre-unfolded prototype matrix (T5 slots in here
  — see T5 for why peel does NOT take sigma/pi handles).

### Selection policy — pinned

**abs-quotient selection + signed coefficient** (the design's own
idiom-reuse language): `best = sims.abs().argmax()`, coefficient
`coeff = sims[best]` (real magnitude, generalizing `_snap_content`'s
±1 `pol` — Spaces.py:2046-2050), subtraction
`residual = residual - coeff * W[best]` (replacing the fixed unit
subtraction at :2370). Rationale: x and −x share an axis (a row and its
exclusion are ONE reference; the sign lives on the symbol scalar, design
§1.5); raw signed argmax would select a *different row* for −u. This
matches `_snap_content` (Spaces.py:2046) and the truth-store query
(`TruthLayer.truth_of`, bin/Layers.py:7300 `sims.abs().argmax()` then
signed best) — the codebase's two existing signed-snap idioms.

**Reconcile the sibling in the same change:** `ChunkLayer.reverse`'s
single-peel branch (Language.py:2328-2348) uses raw signed argmax with no
guard — apply the same abs-quotient + signed-coefficient policy there
(one peel semantics, two entry points).

### Edit sites (all verified current)

1. `ChunkLayer.peel` bin/Language.py:2350-2371: drop
   `if float(sims[best]) <= 0.0: break` (:2367); abs-quotient select
   (:2366); coeff subtraction (:2370); emit `(row, coeff)` pairs; add the
   eps stop. NOTE: Language.py:2366 is NOT the todo.md CPU-Generator fix
   site — that fix drifted to :2574/:2578; coincidental line collision.
2. `factor_percept` (staticmethod on ConceptualSpace, def
   bin/Spaces.py:16141): drop `q = percept.clamp(min=0)` (:16161) and
   `sims.clamp(min=0)` (:16162). **This is a contract change to the
   callosum crossing**, not a local tweak — consumer chain
   `_factor_crossing` (:16178, call at :16205) → `bind_streams` (:16263);
   the docstring frames negatives as "structurally invisible by
   construction" — rewrite that prose deliberately.
3. The order-0 nonneg clamp. **The design's instruction names a parameter
   that does not exist**: `cs_snap_order0` (def Spaces.py:14447) has
   signature `(self, settled_event, ema=False)` — no nonneg. The flag
   lives on its `source_code_activation` call at :14472-14473
   (`nonneg=True, normalize=True`), and the clamp is
   `return act.clamp(min=0.0) if nonneg else act` at :14428 (inside
   `source_code_activation`, def :14394). Edit the call site (and/or the
   clamp). **Domain consequence:** the output is tanh-wrapped AFTER the
   clamp (:14472), so un-clamping moves the `cs_forward_content` source
   term from [0,1) to (−1,1) — a live-path behavior change on the CS wave
   seed, not dead-code cleanup.

### Breakage list (update in the same change, honestly)

- test/test_union_difference_ops.py:122-124 — the ONLY existing peel
  consumer; asserts on the bare index list. Preserve the multiset-exact
  assertion in coeff form (`sorted(r for r,_ in parts) == [2,5,6]`,
  coeffs ≈ 1.0 on the near-orthogonal signed store).
- test/test_interface_factoring.py:106 — pins all-negative-percept →
  a == 0 (the factor_percept clamp contract). Rewrite to the signed
  contract.
- test/test_cs_sparse_weights.py:118-128 — pins source_code_activation's
  nonneg-by-default (:118 explicitly nonneg=False, :128 default-True).

### New test + regression

New signed-peel unit test: an exclusion (negative-coefficient operand,
e.g. whole = W[2] − 0.7·W[5]) is recovered as (5, ≈−0.7); anti-aligned
row selects the SAME row with negative coeff (the abs-quotient property).
Mandatory regression after every T3 edit (live meronomy path, §0):
`RUN_SLOW=1 … pytest test/test_reconstruction_roundtrip.py -q` (21/21,
incl. the S2 round-trip at :577) + test/test_blind_decode.py (5 pass +
1 xfail). Full suite at the gate. **STOP for commit.**

## T4 — Symbol sparsity + the rank-ordered soft-then-hard L0

### The penalty target, corrected (the design's biggest wording drift)

**There is no dense concept→symbol weight matrix.** The signed activation
`a` is the K-step wave state: `cs_forward_content` (def Spaces.py:14491;
`atoms = F.softplus(dictionary)` :14510; `content = a.t().unsqueeze(-1) *
atoms.unsqueeze(0)` :14521) seeds a⁰ = a₀ (order-0 snap, zero-padded,
:14512-14513) and iterates `a_next = ly.wave_step(a, s)` (:14515-14519,
K = symbolicOrder, tanh ⇒ signed). `ly` is the shared **AttentionLayer**
(`SparseLayer` subclass, bin/Layers.py:4708, `[N × N+1]` over the concept
inventory + one trailing bias/EVERYTHING column; roles retired,
self-edges forbidden :4720). **The only learnable weight is
`SparseLayer.values` — a flat `[nnz]` nn.Parameter (declared
Layers.py:4433, allocated in `_grow_values` :4479) with host-side COO
structure (`_rows`/`_cols`/`_index` :4430-4433; device index tensors via
`_indices(device)` :4546-4554).** *(All anchors in T4–T6 are pre-T1d; if
T1d landed first, `AttentionLayer`→`ConceptualAttentionLayer`, `.values`
and the COO structure stay inherited from the generic `SparseLayer` base,
and the `_constituents`/`row_is_identity` record store now lives on
`ConceptualAttentionLayer`. Navigate by name.)*

So reword the design's "each concept→symbol row" to: **each concept ROW
of in-edges over the inventory columns.** The N columns are concepts; the
concept↔symbol 1:1 is the row-alignment convention (symbol i IS the
signed view of concept i — `SymbolSpace.forward_concept_to_symbol`,
bin/Language.py:12645-12668), so a row's in-edge weights ARE its
definition over symbols in the design's sense.

### Mechanism

1. Scatter `values` to a dense `[N, N+1]` view via the cached
   `_indices(device)` long tensors (cheap at current nnz; do NOT iterate
   the host dict per step — `hebbian_strengthen_row`'s dict loop
   (Layers.py:4696-4706) is no_grad/rare-event only).
2. **Mask the trailing EVERYTHING-pole column (col == N) out of the
   definition-size count** — PINNED-BY-PLAN, precedent:
   `cs_groundedness_probe` masks it ("a standing axiom, not perception",
   Spaces.py:14536+). Self-edges never appear (structurally forbidden).
3. Per row: `sorted_w = torch.sort(w_row.abs(), descending=True).values`;
   exempt ranks 1..`definitionFreeSize` (=2, genus + differentia); penalty
   `λ * Σ_{r≥3} sorted_w[r]` — shrinkage lands only on marginal symbols
   while the core two stay unpenalized at full ±1 (the rank-ordered
   soft-L0 of design §5; `torch.sort` backprops through values —
   piecewise-constant permutation, same autograd character as topk/
   max-pool, consistent with the repo's STE idioms; no soft-sort utility
   exists or is needed). Optionally `(n−2)`-weighted per-rank ramp if the
   flat-λ form under-presses — λ sweep decides (see bars).
4. Hard cap: definitions cannot exceed `stmCapacity` — **enforced by the
   decode peel's cap, which is NEW wiring** (T3's `max_parts` threaded
   from the published `model.conceptualSpace.stm_capacity`,
   Spaces.py:12826; knob: model.xsd:974 in conceptualSpaceType, parse
   :12803-12808 default 8). Not by penalty.

### Lifecycle hazards (mandatory)

`SparseLayer.values` is **reallocated as a fresh nn.Parameter on growth
(`_grow_values` :4479) and pruning (`remove_edges` :4540), and is None on
an empty layer.** Compute the penalty inside the loss from the live
attribute every step (then no optimizer-group re-registration problem);
guard `values is None`. Keep the CS forward a static unroll (no
data-dependent control flow — export safety, SparseLayer docstring
Layers.py:4415-4417); the penalty is loss-side, so this is free.

### Wiring (verified templates)

- Loss insertion: the **gate_l1 block is the exact template**
  (bin/Models.py:4607-4620): lambda-gated, default 0.0 ⇒ byte-identical
  off, `totalLoss = totalLoss + term` + `TheError.add(name, term,
  weight=λ, space="ConceptualSpace", category="reg")`, after
  `self.loss.total` (:4488).
- Knobs (neither exists yet — zero grep hits): `definitionFreeSize`
  (default 2) belongs in **conceptualSpaceType** next to stmCapacity (the
  capacity it pairs with — don't split one concept across sections);
  the λ knob (e.g. `definitionSparsityScale`, nonNegativeFloat, default
  0.0) follows the conceptualSimilarityScale template (model.xsd:740 +
  `_t(...)` read at Models.py:1014-1015 + gated additive term).
- The ε dead-zone (design §2.1) is read-time support cleanup ONLY
  (|a| ≤ ε ⇒ don't-care at the peel/support stage — T3's
  PEEL_SUPPORT_EPS); it is NOT part of the training penalty.

**Verify (the §3 bars, measured not asserted):** pairwise-cosine spread
of the CS inventory ↑ (the cone dissolves — baseline the xor store's
pairwise cos ∈ [0.267, 0.827] first); definition-size histogram
concentrates at ≤2 with a tail to 8; terminal residual → 0 on expressed
ideas; S2 round-trip stays green. Sweep λ against these bars (the one
empirical tune; start 1e-3 / 1e-2 / 1e-1). Full suite; **STOP for
commit.**

## T5 — Order-k membership unfold

### Gate, scoped exactly (do NOT block on the whole todo)

The "make abstraction order canonical" todo (todo.md:72-77) gates T5 only
in this subset:

(a) **Live `record_fold` stamping.** Verified state: the table machinery
exists exactly as designed (`Codebook.ramsification` None-init
Spaces.py:2619; `enable_ramsification` :3038; `record_fold` :3060;
`abstraction_order` :3080; `invert_ramsified` :3090) but `record_fold`
has EXACTLY ONE live caller — higher-order minting in
`WholeSpace.maybe_raise_order` (Spaces.py:18415, stamps FOLD_SIGMA on the
minted row) — and `invert_ramsified` has ZERO callers; **T5 is its first
consumer.** The unstamped sites to cover: the mint sites
`_maybe_autobind_meta` (:13691-13692), `_autobind_word_wholes` (:13844),
`insert_meta` (:18171), and the subsymbolic pump loop itself
(Models.py ~6684-6724) — **the PS codebook (PS.subspace.what) is
ALLOCATED a table at Models.py:6202-6205 but NOTHING stamps it**, so
every PS row reads order 0. Stamp on create/select/raise/rewrite through
a σ/π pass, per the todo's live-stamping bullet. T5 landing this
DISCHARGES that bullet; the todo's other bullets (universal allocation /
flag removal, checkpoint persistence, explicit-constraint retraining
consumer, full contract-test battery) STAY OPEN — record that in todo.md
when closing T5.

(b) **The table allocated on the T5 config.** Allocation is gated
`<mereologyRaise>` (Models.py:6186-6205; default False ⇒ table None ⇒
byte-identical). **MM_20M_grammar.xml does NOT set it.** Either enable
`<mereologyRaise>` on the T5/T6 config or land the universal-allocation
todo bullet; the former is the narrow move.

**Silent-degradation guard (mandatory):** with the table None everything
no-ops green — the design's own §2.3 warning. T5's test MUST assert
`cb.ramsification is not None` before trusting any second-lobe result.

### Mechanism — precompute, don't thread handles

`invert_ramsified` (Spaces.py:3090-3121) is per-row and RAISES without
the σ/π layer handles; peel takes only `(whole, basis)` with getW
duck-typing — it has no Codebook access and no handles. Naive per-
iteration unfolding is O(V × max_order) reverse ops per peel step.
**PINNED-BY-PLAN: precompute an unfolded prototype matrix per
abstraction-order class** (group rows by `fold_sequence`, batch-apply the
recorded inversions once per peel call — the owning space, which HAS the
handles and the Codebook, builds it and passes it via T3's `prototypes`
parameter). Use `Codebook.abstraction_order` (fold-count, :3080) — NOT
the concept allocator's `order_of` (Spaces.py:14559-14566; symbolic
recursion bookkeeping, a different "order").

**Alignment hazard:** `Codebook.insert`/`remove` (Spaces.py:3887/3903)
rebuild W WITHOUT resizing the ramsification table, and `record_fold`'s
index clamp (:3069) masks misalignment instead of failing. Add an
alignment assert (table rows == W rows) at stamp/read time; route growth
through `grow_to` (which DOES resize index-aligned, :3257-3263).

**Persistence: accepted open.** The table is a plain attr (not in
state_dict, by design — Spaces.py:2615-2618) and is re-zeroed at build
(Models.py:6204-6205) — after checkpoint reload all stamps are lost and
T5 degrades to order-0 SILENTLY. T5 accepts in-session-only correctness;
the persist-or-sidecar decision stays with the open todo bullet. Say so
in the T5 test comments.

**Verify:** unit — an order-k region member (second-lobe: a probe that an
order-0 inner product misses) scores correctly after the unfold; the
assert-table-non-None guard; existing test_ramsification_table.py
(:102-121 already round-trips invert_ramsified with real σ/π.reverse)
stays green; full suite. **STOP for commit.**

## T6 — Frontier + typed conceptual definition

### Name the right stores (three "relation tables" exist — two are wrong)

- **THE table for T6 = the ConceptAllocator** (bin/Layers.py:4730) over
  the shared square AttentionLayer store: a row = a global concept id
  with role-tagged (part/whole, ref) constituent records
  (`SparseLayer._constituents` :4448-4452, API :4603-4659;
  `row_is_identity` :4646-4649 is the 1:1 part↔whole tie). Rows minted
  via `new_concept`/`relate` (Spaces.py:14030-14044, `relate_idx[key] =
  sid` :14043), `reify_concept`, `singleton_concept`,
  `synthesize_higher_order`, `conceptualize_chain`,
  `assert_concept_relation` (:14870+).
- The concept↔symbol 1:1 is the ROW-ALIGNMENT convention
  (`forward_concept_to_symbol`, Language.py:12645), not table rows.
- **NOT this one:** the `insert_relation` predicate/idea rows
  (Spaces.py:18182, iterated by `_iter_relation_rows` :15366) — that is
  reasoning.py's store. Wiring it here is the wrong table.

### Reuse the two existing frontier implementations — do not write a third

- **Concept side:** `ConceptualSpace.prune_concept_links`
  (Spaces.py:14779) — docstring literally "keep the MAXIMAL part and the
  MINIMAL whole among each concept's links"; drops the EVERYTHING pole
  once a concrete whole exists (:14793-14798), drops taxonomy-ancestor
  wholes via `_whole_ancestors` (:14764-14777), drops covered raw parts
  (:14807-14815).
- **Percept side:** `RunStructureLayer.tightest_container`
  (bin/Layers.py:3165-3183 — smallest strict container per byte-span =
  the minimal covering whole; containment mask :3154-3155; owner
  `WholeSpace.run_structure` Spaces.py:17033).

T6 composes these over the peel's `(row, coeff)` support: keep maximal
parts (inclusions not implied by others) + minimal wholes (tightest
covers); negative-coeff members become exclusions. Emit the **typed
definition**: head = minimal covering whole; modifiers = surviving
maximal parts; exclusions = negative-polarity members.

### Coverage decision (the "strength only" gate)

The promotion plan (doc/plans/2026-07-04-attention-to-relation-promotion.md)
is **0/7 implemented** — plan-only. What EXISTS is exactly the "strength
only" substrate: relation-table rows + Hebbian-bumped edge scalars
(`hebbian_strengthen_row` Layers.py:4696-4705 via `_hebbian_strengthen`
Spaces.py:14915-14927; assertion weights `_set_concept_edge_value`
:14842). BUT all population sites are behind `<mereologyRaise>` and
**MM_20M_grammar.xml does not set it ⇒ the concept-side table is EMPTY on
the decode config** — concept-side pruning would no-op end-to-end.
**Decision (PINNED-BY-PLAN):** enable `<mereologyRaise>` on the decode
config for T6 (same flip T5 needs), and accept that sparse coverage ⇒
verbose-not-wrong definitions (design §2.4). Promotion lands later as a
coverage feeder (it also CONSUMES the honest residual T2/T3 create —
design §3.3); do not block on it.

### Grammar compression leg — two traps

- **No literal `NP → ADJ + NP` production exists.** All shipped grammars
  are operator-role form; ADJ == INTERSECTION (data/complete.grammar:35;
  compose rule :122; generate/reverse rule :165); negation = not/non
  (:110-112 compose, :160-161 generate; `NegationLayer`
  bin/Layers.py:3280). role_collapsed.grammar no longer exists in data/.
  "The NP→ADJ+NP peel" = the `intersection.reverse` generate rule.
- **Fail-loud basis requirement:** post-Gate-S1, the intersection/union/
  conjunction/disjunction reverses REQUIRE a codebook basis (recommender);
  the no-basis lattice tails RAISE (complete.grammar header :52-53).
  Compressing the typed definition to a surface without threading the
  basis ERRORS, it doesn't degrade — plumb the basis where the generate
  path invokes the reverses.

### Verification — two bars, stated honestly

The design's cited meter (Method-2 surface match = serial-plan Task 4) is
**UNSTARTED, and its prerequisite Task 3 (NULL-word emit/recognize, needed
for blind length round-trip) is also unstarted** — both outside this plan
(doc/plans/2026-07-04-serial-derivation-reconstruction-execution.md:
Tasks 0-2 + S2 DONE/GREEN via Method-1 leaves replay `_reverse_method1_
leaves` Models.py:9403/4367; :505-508 Task 3 unstarted). So:

1. **Within-plan bar (T6's own gate):** typed-definition unit checks on
   constructed cases — head/modifiers/exclusions correct where the
   frontier inputs are staged; peel→frontier→typed-definition on a
   mereologyRaise config.
2. **The end-to-end meter (joint close with serial Task 4):** stand up
   the Task-4 harness EARLY as pure measurement if cheap
   (`reconstruct_from_idea=True` scored on surfaces against Method-1
   teacher — Models.py `reconstruct_from_idea` gate ~:798,
   `_chart_generate_from_stm` ~:8299, bin/recon_bench.py,
   test/test_reconstruction_roundtrip.py) — it just reports a
   ceiling-bounded score that T3–T6 raise. Full blind round-trip waits on
   serial Task 3; T6 does NOT block on it.

Decode-seed anchors (what T6's pipeline consumes): the reduced root S is
parked at `self._stm_single_S` (Models.py:8194, from
`_stm_reduce_to_single_S` :8190); the serial tensor-reverse seed is
`snap[:, :1, :]` (root at STM slot 0) in `_reconstruction_seed`
(:3505/:3531); Method-1 leaf slab `_stm_pre_reduce_slab` (:8159-8161).

Full suite; **STOP for commit.**

## Dovetail map (todo.md ↔ this plan)

| todo.md item | relation | disposition |
|---|---|---|
| :22-32 concepts/snap-contract design+build | SUBSUMED | this plan executes it; mark done when T1–T6 land |
| :54 full-suite status / Gate-4 close | GATES T1 | **done as T0** (this doc records the result) |
| :72-77 abstraction order canonical | GATES T5 (subset) | T5 discharges the live-stamping bullet; allocation/persistence/retraining bullets stay open |
| :36-42 serial Task 4 (Method-2 bar) | GATED_BY T3-T6 / is T6's meter | harness can stand up early (pure measurement); full round-trip also waits on serial Task 3 (NULL-word) |
| :85 attention-to-relation promotion | soft-GATES T6 (strength only) | 0/7 built; T6 correct-but-verbose without it; sequence after T3 (it consumes the honest residual) |
| :50 COMPILE/PERF + NULL-word | INDEPENDENT | zero file overlap with T3 ("peel" absent from Models.py); owned by serial Task 3.3 |
| :56-59 property_basis hack | DOVETAILS T1c | T1c rewrites its PROSE only; mechanism removal stays a separate deliberate step (anchors drifted: 2599→2762, 6718→6907) |
| :61-64 TruthSet→LTM + design §1.6 two-sided evidence | INDEPENDENT | bundle §1.6 (± axes feeding `act = pos−neg`, Spaces.py:6477 — note the twin site :6114 in `_compute_active`) into the TruthSet-to-LTM session |
| :67-70 masked semantic recon config | DOVETAILS T4 | sequence AFTER T4 or re-tune λ under the changed loss mix; uses mereologyRaise like T5/T6 |
| :14-20 Gate-B xfail | INDEPENDENT | expected xfail in every gate run |
| :3-12 query-tool integration; :44 .where recovery (re-audit — recon_bench.py:69 now real); :46/:48; :52 GPU; :79 FCA doc; :81-83 relation-table notes | INDEPENDENT | :81-83 is consistent with T6's table reading (one entry = one concept-symbol binding; repeated entries with the same concept index = set definition; recursive relation concepts = vine); FCA doc can cite design §1.3 |

## Fidelity bars (design §3 — measured, not asserted)

Instrument once (T2's probe), read at every task gate: pairwise-cosine
spread per inventory (baseline first: xor store [0.267, 0.827]); radial
spread by composition depth; peel-termination margin (|coeff| at stop vs
eps); terminal residual on expressed ideas; definition-size histogram
(T4); Method-2 surface match (end-to-end, once the harness exists).
Persistent structured residual → the promotion plan's mint signal; small
never-vanishing residual → crowding (spread rows).

## Stop-point summary

| stop | contents | gate |
|---|---|---|
| T0 | join→union in 6 inline grammars | 3 red tests green; full suite 3008/0 recorded |
| T1a | insert_symbol→insert_whole (+strings, test naming, SymbolFirewall.md) | full suite == T0 count |
| T1b | \bnObj\b→nIdeas (nObjects protected); ObjectSubSpace→IdeaSubSpace | full suite == T0 count |
| T1d | split concept block out of SparseLayer → ConceptualAttentionLayer (structural, behavior-preserving); fix ConceptAllocator.layer both branches; split the test file | full suite == T0 count |
| T2 | <initScale> (wholeSpaceType; create-kwarg threading; writer guards/probes) + T1c prose if not earlier | new depth-probe + suite |
| T3 | signed peel (final contract; abs-quotient; eps stop) + factor_percept + order-0 unclamp; 3 pinned tests updated | signed-peel unit + S2 regression + suite |
| T4 | rank-ordered L0 on SparseLayer.values rows (pole masked); definitionFreeSize + λ knobs; stmCapacity→max_parts | §3 bars + λ sweep + suite |
| T5 | live record_fold stamping; mereologyRaise on decode config; per-order-class prototype unfold via peel `prototypes=` | second-lobe unit (table non-None asserted) + suite |
| T6 | frontier reuse (prune_concept_links + tightest_container) over peel support; typed definition; basis threading into generate reverses | typed-definition units; optional early Task-4 meter |
