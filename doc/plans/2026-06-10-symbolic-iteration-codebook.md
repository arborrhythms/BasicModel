# Symbolic-iteration codebook on the CS→SS leg

**Status: NOT STARTED — this is the next iteration's plan.**
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
