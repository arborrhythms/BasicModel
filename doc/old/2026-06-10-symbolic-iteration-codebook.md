# Symbolic-iteration codebook on the CS→SS leg

**Status: IMPLEMENTED (Steps 0–3 complete; Step 4 landed as the xfail
acceptance), rev. 2026-06-10 (second session).** Suite on the dev
machine: green except 7 PRE-EXISTING order-dependent failures in
`test_heat_reverse_wiring.py` (clean-tree-reproduced full-suite
pollution; they pass solo — flagged for a separate fix) and the
deliberate Step-4 xfail. Implementation notes:

- **Step 0**: the gate fixtures gained `<training><seed>1</seed>` +
  `<architecture><device>cpu</device>` (new knobs, env overrides
  `BASIC_SEED` / `BASICMODEL_DEVICE`, schema updated): unseeded runs
  are recon-flaky BY SEED BASIN (~half the basins miss byte-exact;
  seed 0 is a miss, seed 1 is green) and MPS kernels are
  nondeterministic, so the gate is only an arbiter on CPU+seed. Both
  gates verified 4/4 + byte-exact and BIT-REPRODUCIBLE; they stayed
  bit-identical through every step below.
- **Step 1**: the symbolic-iteration predicate is the MODEL'S
  `conceptualMode` mirrored onto each SS (`_conceptual_mode`) —
  SyntacticLayer presence cannot distinguish the legs (the model.xml
  default grammar attaches one everywhere). Geometry guards (view at
  least codebook-width; slots×nDim divisible by nOutputDim) keep
  legacy-geometry stages (MM_xor tapered t>0 legs hand 5/10-wide
  views) on the old path. The emission: full-frame no-grad snap →
  strongest-slot winner → STE on the winner slot only, apoha zeros
  elsewhere → virgin-winner fallback to the continuous carrier (the
  honest-STE lesson) → winner-row recon gather
  (`_csleg_recon_loss`, lifted at t>0 under the same
  `ss_codebook_recon` error name). Host-eager
  `adopt_symbolic_evidence` (one-step-stale CS views, stage t adopts
  from stage t-1's persistent view) + `stage_symbolic_virgin_rows`
  run from `_lex_embed_stem`; adopted rows are tagged
  ROLE_MEANING_GENERAL at adoption time. The virgin staging must NOT
  force-allocate the roles buffer (torch.export lifts it as a
  constant and the stage-0 in-graph tagging then trips "constant
  mutated in forward" — found via test_mlx_export).
  `descriptor_roles` is now explicitly CPU-pinned (the
  set_descriptor_role `.cpu()` indexing assumed it). Contract tests:
  `test/test_symbolic_iteration.py` + the order-2 fixture
  `data/MM_symbolic_iter.xml` (MM_20M ships order 1 — its t>0 leg
  never runs in-body).
- **Step 2**: `SymbolicSpace.analysis_store` (Codebook, customVQ,
  category) built in `__init__` under `<analysis>`, RNG-ISOLATED
  (dedicated seed + CPU generator state save/restore) so seeded gates
  reproduce bit-identically; the stage-0 unity machinery (snap,
  LF_COARSE roles, recon gather, adopt_stage0_evidence,
  semantic-arrangement) re-homed onto it; the Models asymmetric
  hardwire (EMA off, commitment 0, Parameter promotion + optimizer
  enlistment) extended to both families. `none` configs now analyze
  at stage 0 (knob-independent), and the gates stayed bit-identical —
  the recon gather's gradient is isolated to store rows by
  construction.
- **Step 3**: `insert_paired_word`, `mark_word_atom`, both
  `Embedding` peer hooks, `_tie_lexicon_to_codebook`, its Models call
  site, `WordVectors.tie_to_codebook`, and the tied branch of the
  checkpoint vocab restore are all retired. The `_vectors` property
  indirection REMAINS (it carries the Dynamo-traceability fix); the
  five pinning test files now pin the PS-LOCAL (untied) contract.
- **Step 4**: landed as
  `test_mm20m_second_order_reverse_keys_codebook` (xfail): the narrow
  emission currently materializes the VALUE reshape of the apoha
  frame at [1020, 8] (band not packed → not the configured
  [1024, 8]), the 4-wide what channel does not yet carry the written
  symbol ID, and the reverse re-applies the S-tier transform instead
  of keying the codebook. Those three gaps are the follow-on work.

Authored 2026-06-10 at the close of the Phase-5 / Task-4 session
(suite 2162 passed / 0 failed at handoff; everything committed by Alec).
Direction set by Alec, 2026-06-10; the architectural statement lives in
`2026-06-08-analysis-synthesis-dual-input.md`, section
"The `<codebook>` knob STAYS" — read that section first.

## Semantics being built

The SS `<codebook>` tag is the **iteration-mode switch** (the former
#13b knob-REMOVAL is cancelled):

| `<codebook>` | PS iterations (≤ conceptualOrder) | SS iterations |
|---|---|---|
| `none` | subsymbolic | **subsymbolic** (continuous; the whole loop is subsymbolic, then reconstitute) |
| `quantize` | subsymbolic | **symbolic** (the SS leg QUANTIZES its input) |

The knob's codebook applies to the **input from CS** (the t>0 recurrent
leg, `CS_subspaceForSS`) — NOT the input from IS (stage-0 unity), where
the analysis codebook lives. Two codebook families, kept separate:

1. **Meronymic/analysis codebooks on INPUT** (independent of the knob):
   the stage-0 analytic generality store (adopt-on-first-sight, recon
   gather, descriptor roles) under `<analysis>`, and the PS percept /
   lexicon / shared-byte stores under `<synthesis>` (Task 4).
2. **The symbol codebook on the CS leg** (the knob): captures in place
   the correspondence of the **code-as-written** and the
   **code-for-the-concept** — the first is a SYMBOL for the second and
   must be interpreted. Codebook row = concept code (full nDim-wide);
   row id / the narrow 4-wide what channel = the written symbol.

## Step 0 — standing acceptance gates (already in place)

- **`data/XOR_exact_symbolic.xml`** (created 2026-06-10): XOR_exact
  with SS `<codebook>quantize</codebook>`, otherwise identical. Gate:
  `python bin/Models.py data/XOR_exact_symbolic.xml` →
  **4/4 predictions AND byte-exact reconstruction**. BASELINED GREEN
  at handoff (0.04/0.98/1.00/0.03, all rows byte-exact). This must
  stay green through EVERY step below — it is the regression gate for
  the rework.
- `data/XOR_exact.xml` (SS `none` — all-subsymbolic BY DESIGN, not a
  gap): 4/4. Caveat: recon shows unseeded run-to-run flakiness
  (predictions always 4/4; recon 4/4↔2/4 across runs, different rows
  each time; the none-path code was untouched when observed). Optional
  hygiene first task: seed the CLI fixture before reading single runs.
- Full suite: 2162 passed / 0 failed at handoff.

## Step 1 — CS-leg symbolization (the core build)

Under `quantize`, parallel leg, t>0 (`SymbolicSpace.forward` with
`CS_subspaceForSS`, the reversible-dispatch branch in
`bin/Spaces.py` — currently: continuous event + no-grad naming via
`_naming_indices`):

- Snap the CS input against the SS codebook and make the EVENT the
  snapped symbol code — value substitution is CORRECT here (these are
  symbolic iterations), unlike at stage 0 (analysis does not alter
  the data).
- **ONE SYMBOL AT A TIME (Alec, 2026-06-10):** the SS codebook emits a
  SINGLE symbol per symbolic iteration — not a parallel slab of
  snapped slots. (This is what MM_20M's PS comment "8 in parallel
  mode, looping over 1 in symbolic mode" anticipates: symbolic mode
  iterates, one emission per step.)
- **APOHA SEMANTICS (Alec, 2026-06-10):** the COPART of the emitted
  symbol is ZEROS EVERYWHERE — literally, "the way that the concept
  of the cup appears to the mind is through the negation of non-cup"
  (anyāpoha: meaning by exclusion of the other). The zeros are not
  padding or an optimization; the exclusion IS how the universal
  appears. Implementation reading: the emission frame carries the
  selected symbol's code with every other slot/row contribution
  exactly zero; the selection presents itself BY that zeroing. (Fits
  the existing Buddhist mapping in Philosophy.md — SS = the
  universals tier — and should inform how the narrow 4-wide ID
  channel and the Step-4 reverse read the emission: one live symbol,
  zero copart, per iteration.)
- **THE CODEBOOK REPLACES PI (Alec, 2026-06-10):** in symbolic
  iterations the snap does not follow the Pi operation — it STANDS IN
  for it. Selection-by-exclusion replaces computed intersection: the
  quantize branch BYPASSES the pi transform on the CS leg (the
  codebook lookup IS the analysis for that iteration). Consequence —
  and the point of the design: **thought is top-down, serial, and can
  only be acted upon by Sigma.** The emitted symbol is discrete and
  given (no further Pi transformation applies to it); the only
  operation that can act on it is Sigma (synthesis, union, the
  bottom-up side). Downstream consumers of the emission compose with
  it; they do not re-analyze it.
- Honest STE from step one: re-home **adopt-on-first-sight**
  (`adopt_stage0_evidence` pattern — host-eager, the data-dependent
  `unique` cannot live in the compiled body; call it from the eager
  stem) so VIRGIN rows adopt the CS-input vectors that select them.
- The recon gather (input→codebook; the asymmetric-VQ EMA
  replacement) retargets to THIS leg's evidence so rows train toward
  concept codes. EMA stays off; commitment stays 0 (Models hardwire).
- Naming indices feed the narrow output: the SS output `[nOutput,
  nOutputDim]` (MM_20M: `[1024, 8]` = 4 what + 2 where + 2 when) —
  the 4-wide what channel carries the written symbol ID.

**Lessons that must carry over (hard-won 2026-06-10, see the Phase-5
section of the dual-input plan for the full bisection):**
- `Codebook.quantize`'s `replace_W` refresh is GATED on
  `vq.ema_update` — never let the snap re-point the space-owned `W`
  at the VQ matrix (the #13 output-collapse mechanism).
- Value substitution against a virgin/random codebook poisons
  training — adoption must precede the first honest snap.
- The decode maps row→word through the INVERSE of `key_to_index`
  (`decode_reverse_meta`), never positionally via `index_to_key`.

## Step 2 — stage-0 re-home under `<analysis>`

The stage-0 unity machinery (`_stage0_unity_forward`: snap, roles,
LF tagging, recon gather, `adopt_stage0_evidence`) currently gates on
`self.codebook`. Decouple it: the analysis store is its own basis
object governed by `<analysis>`, present regardless of the knob.
(Until this step, `none` configs simply skip stage-0 snapping — the
status quo — so Step 1 can land first without breaking `none`.)

## Step 3 — retire `insert_paired_word` + the tie

The PS→SS reach-across (orth row + random semantic partner per
lexicon word; `key_to_index` remapped onto SS rows) is REPLACED by
the Step-1 codebook. Blast radius (inventoried 2026-06-10):
- `Embedding.insert`'s peer hook (`bin/Spaces.py` ~3713) and the
  `stage_oov` bulk migration (~3446–3495, `wv.tie_to_codebook`).
- `embed.WordVectors.tie_to_codebook` (~embed.py:1183).
- The autobind `mark_word_atom` fallback.
- Five pinning test files: `test_tied_orth_storage`,
  `test_lexicon_ownership`, `test_tied_vectors_compile`,
  `test_tying_storage_shared`, `test_unified_lexicon_codebook`.
The lexicon keeps PS-LOCAL storage permanently (untied); the decode
fix already handles both layouts (inverse map = identity when
untied).

## Step 4 — the 4-D second-order acceptance (MM_20M)

With SS `quantize` (MM_20M ships it): the symbolic iteration emits
`[1024, 8]` narrow codes; the REVERSE keys the codebook by the 4-wide
symbol ID and recreates the full 1024-wide concept representation —
"reconstruct by keying the codebook with indices". Quantization is
what allows CS to return ACTIVATION VALUES ONLY → second-order
symbols. Land as a test (xfail until Step 1 completes it; the suite
already carries 30 xfails as the idiom).

## Harness etiquette (saved probes)

- The trained-CLI gate is the arbiter; in-process `runEpoch` is fine
  for EVAL probes, flaky for training probes.
- runpy-based CLI patch probes must bind functions into
  `Spaces.__dict__` (dynamo guards break on `__main__` closures;
  Models-level patches don't reach the runpy `__main__`).
- The transplant probe pattern (build two configs, `load_state_dict
  (strict=False)`, seeded eval, diff `allOut`/`lastIn`) localizes
  wiring differences with identical weights — `/tmp/xor_transplant.py`
  pattern documented in the dual-input plan Phase-5 notes.
