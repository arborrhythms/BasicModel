# `.where`/`.when` Encoding Pass — Execution Plan (DRAFT for Alec review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
> **Alec does ALL git writes** — GATES are his commit points. Design doc
> (DRAFT, review pending):
> [2026-07-04-where-when-encoding-design.md](2026-07-04-where-when-encoding-design.md).
> **NO task begins until Alec approves design + this plan.**

**Goal:** `.where` period decoupled from nObjects (config-derived, default
8192) with an overflow warning; `.when` re-encoded as 4 dims (LF start
pair + HF sin start-fine + HF sin duration, exact clock remains the
addressing side-band); band-width plumbing throughout; the Gate-2b BLIND
round-trip bar (tiling re-derived from the band, scaffold demoted to
debug); one coordinated global re-baseline.

**Defaults adopted where the design left options (flag at review):**
Q1 = Option C hybrid (clock addresses; band = the decided 4-dim shape).
Q2 = `.where` period config-derived (`<wherePeriod>`, default 8192).
Q3/HF = shared short HF per the decided shape (variant (a) recorded).
Q4 = the blind bar STANDS BESIDE the scaffold bar (scaffold pin kept as
the content-identity regression).

**House rules for every task:** no git writes; comments one-liners;
targeted pytest (`PYTHONPATH=test:bin BASICMODEL_DEVICE=cpu
MODEL_COMPILE=eager .venv/bin/python -m pytest ...`); `make test` only at
the final gate on a quiet tree; Inf/NaN fail loud; LaTeX in docs; probes
in the scratchpad; value-pins updated honestly with each change flagged.

**Key sites (verified this session; re-verify lines before editing):**
`QuadratureEncoding` bin/Spaces.py:728; `_bracket_encode` :860;
`WhereEncoding` :913; `WhenRangeEncoding` :1052 (+ alias :1173); period
build seam :9383-9391; runtime guards :997, :12038; `.where` stamp site
`_embed_radix` :10404; decode `_decode_where_offset` :4034,
`_decode_radix_meta` :11912; `canonical_shape` bin/Layers.py:13611;
`ModelLoss.compute` per-call widths :13633; `_reverse_event_loss`
bin/Models.py:2769; blind-decision + probe records in the reconstruction
plan's EXECUTION NOTES.

---

## Task 0 — Baselines + the correlation micro-probe (measurement only)

- [ ] **0.1** Capture pre-change harness records (seed 0, epochs 3,
  cpu/eager) for xor / legacy / grammar / idempotent at the CURRENT tree
  (post silent-band-twins fix — use its re-baselined numbers) into
  `output/` and quote the table in EXECUTION NOTES as "pre-encoding
  baseline".
- [ ] **0.2** Correlation micro-probe (scratchpad, extends
  `probe_where_precision.py`): stamp TWO quadrature pairs into one event
  through forward+reverse; report the inter-band error correlation
  matrix. If common-mode correlation is high ($> 0.7$), note in the
  design doc that the duration channel inherits the differencing bonus;
  no design change either way (informational for decode weighting).

## Task 1 — `.where` period decoupling + overflow warning

**Files:** bin/Spaces.py (build seam :9383; WhereEncoding ctor), data/model.xsd
(`<wherePeriod>`), bin/Models.py (config parse), test/test_where_encoding_period.py (new)

- [ ] **1.1 Failing tests first** (new file): (a) a config with
  `<wherePeriod>8192</wherePeriod>` builds a WhereEncoding with
  `maxVal == 8192` regardless of nObjects; (b) absent tag → default 8192;
  (c) an input longer than the period triggers ONE RuntimeWarning naming
  config, length, period, and "increase <wherePeriod>" — and the period
  is raised to fit (never silent aliasing); (d) the runtime asserts
  (:997, :12038) still hold.
- [ ] **1.2** Implement: parse `<wherePeriod>` (XSD comment, no `--`);
  the build seam uses it instead of nObjects, keeping raise-to-fit +
  the new warn-once. NOTE: this alone re-scales every `.where` stamp —
  it is re-baseline event material; do NOT capture new baselines until
  Task 3 lands (one coordinated re-baseline, not two).
- [ ] **1.3** Codec tests green; `test_where_encoding_recovery` /
  `test_where_bracket` / `test_where_decoupling` / `test_ps_where`
  adapted ONLY where they pin the old period value (flag each).

## Task 2 — `.when` v2: the 4-dim start/duration encoding

**Files:** bin/Spaces.py (`WhenRangeEncoding` :1052 region — new class
`WhenStartDurationEncoding`, alias update :1173), test/test_when_encoding_v2.py (new)

- [ ] **2.1 Failing codec tests first** (new file; pure encode/decode, no
  model): LF pair round-trips onset over the configured horizon
  (`<whenPeriod>`, default $10^6$) at float precision; HF pair
  round-trips the onset residue within one rung ($P_{hf} = P_{lf} /
  \texttt{<whenRungRatio>}$, default ratio 32, safe bound
  $\pi/\Delta\phi \approx 35$ at the dense floor); the LADDER decode
  recovers exact onset when the LF estimate localizes within half an HF
  period (pin the boundary case); `shift_time` rotates BOTH pairs
  coherently (decode after shift == shifted decode); constant norm
  $\sqrt{2}$ pinned exactly; the exact clock (`self.t`, serialization)
  is UNTOUCHED.
- [ ] **2.2** Implement the class: 4 dims `[sin(s·w_lf), cos(s·w_lf),
  sin(s·w_hf), cos(s·w_hf)]` — TWO TRUE QUADRATURE PAIRS, ONE quantity
  (onset), a 2-rung ladder (Alec FINAL 2026-07-04; duration REMOVED
  from the band — it was write-only in the codebase: decode_span has
  zero callers, tense machinery rotates the center only, aspect retired
  at Language.py:3106. Exact durations live in the record store when
  aspect is eventually built). `decode` = atan2 per pair + HF branch
  resolution by LF (rung ratio a config constant, default 32, safe
  bound pi/dphi ~ 35 at the dense floor); returns (start, residue
  diagnostics). `shift_time` = exact phase rotation PER PAIR at its own
  omega. Tense derivation (past/present/future vs `self.t`) keys on the
  LF pair. `nDim = 4`. Endpoint-sum bracketing no longer serves `.when`
  (WhereEncoding keeps it). Codec tests (2.1) updated: LF round-trip
  over the horizon; HF round-trip within a rung; ladder decode exact
  when LF localizes within a half HF period; shift_time preserves both
  phases coherently; constant norm == sqrt(2) pinned exactly.
- [ ] **2.3** Codec tests green; grep `WhenEncoding|WhenRangeEncoding`
  consumers (bin/Language.py:8732 fixture, bin/Models.py imports, tense
  readers) — adapt call sites; consumers that assumed nDim=2 slots get
  the Task-3 layout treatment.

## Task 3 — Band-width plumbing + the coordinated re-baseline (GATE A)

**Files:** bin/Layers.py (`canonical_shape` :13611), bin/Spaces.py (muxed
layout: when slots [-2,-1] → [-4..-1], where shifts to [-6,-5]), bin/Models.py
(`_reverse_event_loss` widths), test/test_reconstruction_roundtrip.py (pins)

- [ ] **3.1** `canonical_shape("InputSpace")` (2,2) → (2,4); trace EVERY
  consumer of the muxed tail (grep the slot arithmetic `[-4`, `[-3`,
  `[-2`, `[-1` in Spaces.py encode/decode/stamp sites) and shift to the
  new layout via ONE shared layout helper (no scattered magic offsets —
  add `Muxed.layout()` or equivalent single source of truth).
- [ ] **3.2** The loss seam passes (nWhere=2, nWhen=4); the fail-loud
  guard (warn-once, never silent what-scaling) covers any path that
  cannot know the widths — the fourth-instance rule.
- [ ] **3.3** THE COORDINATED RE-BASELINE: re-run Task 0.1's four
  configs; record old → new; update every value-pin (list each; the
  ratio pin re-measured; convergence re-measured E={30,38,50} and
  EPOCHS_PINNED re-derived by the standing formula if moved). THE
  scaffold bar must return to green at the (possibly re-pinned) budget.
- [ ] **3.4** Serialized-model incompatibility: confirm old checkpoints
  fail LOUDLY on load (shape mismatch raises, not silent misread);
  note in EXECUTION NOTES. Sweeps: roundtrip file, handoff (7),
  recon_bench, sparse_concept_e2e, where-path four, when/tense tests.

**GATE A (Alec commits):** new encodings + plumbing + re-baseline green.

## Task 4 — Blind decode: Gate 2b's bar (GATE B)

**Files:** bin/Spaces.py (`_decode_radix_meta` :11912 region), bin/recon_bench.py
(blind mode flag), test/test_reconstruction_roundtrip.py

- [ ] **4.1 Failing bar first**: `test_mm20m_xor_blind_roundtrip` —
  scaffold OFF (explicit flag), tiling re-derived from the `.where`
  band: decode coarse claims per active slot → tile hypotheses → snap by
  running-sum consistency + percept byte-sizes (the two-arm association
  unchanged); exact_match == 1.0 at a budget measured post-Task-3
  (expect ≥ the E≈80 byte-exact crossing; pin by the standing formula;
  RUN_SLOW).
- [ ] **4.2** Implement the SCAFFOLD-MASKING CURRICULUM (Alec
  2026-07-04): `scaffolding==true` masks the EXISTING masking-rate
  fraction (the XML maskRate parameter — verify the exact tag,
  `<maskRate>`/training.maskRate, at execution) of the scaffold tiles;
  masked tiles decode BLIND (from the band), unmasked tiles are given.
  mask rate 1.0 == fully blind; the 5c/5d scaffold pin keeps running at
  rate 0.0 as the content-identity regression (Q4 confirmed). Dense
  per-masked-tile feedback is the free-run training signal. Active-slot
  gating for blind tiles derives from the band claims + magnitude floor
  (a real stamp is ~1.0; pads carry none).
- [ ] **4.3** `recon_bench` gains `--blind/--scaffold` (blind default);
  `where_recovery` computed blind. Sweeps + the scaffold-mode pins green.

**GATE B (Alec commits):** blind bar green beside the scaffold pin.

## Task 5 — Docs + notes (GATE C, with `make test`)

- [ ] **5.1** Architecture sec D: correct the period claim (config-derived
  `<wherePeriod>`, default 8192; was wrongly "½·InputSpace", actually
  Σ nVectors pre-change); sec A/B pointers to the blind bar. Spaces.md
  encoding section: the 4-dim `.when` shape + shared-HF caveat + clock
  side-band. WhenRangeEncoding docstring already updated in Task 2.
- [ ] **5.2** EXECUTION NOTES: baselines old→new, correlation-probe
  result, pins ledger, deviations.
- [ ] **5.3** `make doc` green; `make test` on a quiet tree — THE gate.

**GATE C (Alec commits; plan done).**

---

## EXECUTION NOTES (append during execution)

### Task 0 — pre-encoding baselines + correlation micro-probe (2026-07-04, local ArborStudio, cpu/eager, seed 0, epochs 3)

**0.1 Pre-encoding baseline** (bin/recon_bench.py at the post-sync tree,
basicmodel 9492b0e clean; these are THE "old" column for Task 3.3):

| config | output_loss | recon_loss | exact | where_rec | s/epoch |
|---|---|---|---|---|---|
| `MM_20M_xor` | 0.1868378072977066 | 0.01573723368346691 | 0.5 | 1.0 | 0.778 |
| `MM_20M_legacy` | 0.12448841333389282 | 0.07315883040428162 | 0.0 | 0.0 | 0.166 |
| `MM_20M_grammar` | 0.17499999701976776 | 0.002442623721435666 | 0.0 | 0.25 | 1.63 |
| `idempotent` | 0.17494817078113556 | 0.0 | 0.0 | 0.0 | 0.230 |

Records: `output/recon_MM_20M_xor_20260704-102020_58191.json`,
`recon_MM_20M_legacy_20260704-102038_58208.json`,
`recon_MM_20M_grammar_20260704-102048_58214.json`,
`recon_idempotent_20260704-102102_58232.json`. (Values differ from the
2026-07-03 plan's tables — that tree predates the committed post-Gate-3
fixes; this table is the honest pre-encoding zero point.)

**0.2 Correlation micro-probe** (scratchpad
`probe_band_correlation.py`: probe-local `_embed_radix` hook stamps a
per-slot `.when` pair beside the live `.where` stamp — two pairs, one
event, trained forward+reverse, MM_20M_xor regime; 12 active slots):
inter-band PHASE-error correlation $\rho(\Delta\phi_{where},
\Delta\phi_{when}) = 1.000$ — the transport noise is COMMON-MODE
(component matrix: dW.sin/dT.sin 1.000, dW.cos/dT.cos 0.899; |err| med
0.18 vs 0.17 rad on the two bands). Verdict per design addendum: HIGH
(> 0.7) — a difference channel would cancel the shared carrier noise;
noted in the design doc, no design change (informational for decode
weighting).

### Tasks 1–3 — implementation record (2026-07-04, Gate A)

**Task 1 (`.where` period).** `<wherePeriod>` declared (model.xsd
architectureType, default 8192 = 2× the 4096 `inputLength` cap);
construction site (bin/Spaces.py `Space.__init__` whereEncoding) reads it
instead of `architecture.nObjects`; `_WHERE_PERIOD_DEFAULT = 8192` beside
`_WHEN_PERIOD`. The raise-never-lower seam MOVED out of the
Embedding-only branch to ALL synthesis modes (the radix path stamps byte
offsets too and would alias silently — the seam was lexicon-only, a
latent gap) and gained the warn-once (config name, byte length, period,
"increase `<wherePeriod>`"; raise-to-fit retained, string inputs keep the
2× headroom). The `reconstruct_to_buffer` periodicity assert's remedy
now names `<wherePeriod>` (was the retired nObjects coupling). Tests:
`test/test_where_encoding_period.py` (5, TDD red→green; the overflow pin
uses xor's measured 12-byte max, raise 4→24). Where-path sweep 17
green; roundtrip/bench/render 26 green pre-flip.

**Task 2 (`.when` v2 codec).** `WhenStartDurationEncoding`
(bin/Spaces.py, after WhenRangeEncoding): 4 dims
$[\sin(s\omega_{lf}), \cos(s\omega_{lf}), \sin(s\omega_{hf}),
\cos(s\omega_{hf})]$, ONE quantity (onset), 2-rung ladder;
`<whenPeriod>` default $10^6$, `<whenRungRatio>` default 32 (both in
model.xsd); decode = atan2/pair + HF-branch-by-LF, returns (start,
residue-diagnostic); `shift_time` = per-pair rotation at its own
$\omega$; tense keys on the LF pair (`next`/`previous`); `forward`
stamps `encode(self.t)`; exact clock untouched; NO `decode_span`, NO
end/D kwargs (endpoint-sum bracketing retired for `.when` — fail-loud
TypeError, not accept-and-ignore). Float32 hardening: encode folds the
onset into each pair's period pre-multiply (unfolded, $s = 10^6$ cost
~0.08 HF ticks); decode seam-folds $[P-0.5, P) \to [-0.5, 0)$ so integer
onsets round exact at the 0/P seam. Codec tests:
`test/test_when_encoding_v2.py` (11, TDD; boundary case pins BOTH sides
of the half-rung branch bound; one bound honest-widened to the measured
float32 ULP noise near $\pi$, 0.04 ticks). Class name keeps the plan's
4-value lineage (duration since removed from the band) — flagged, per
the plan's own Task-2.2 text.

**Task 3 (plumbing + re-baseline).** `canonical_shape`: all six interior
sections $(2,2) \to (2,4)$; OutputSpace stays $(0,0)$. ONE construction
seam `Spaces.event_when_encoding(n_when)` reads the two knobs — the
Space build AND the grammar tense ops (`Language._event_when_encoding`,
`_WhenOpMixin._when_encoding`) all construct through it (a mismatched
period would silently rotate wrong). `WhenEncoding` alias →
`WhenStartDurationEncoding`. `_EVENT_WHEN_WIDTH`/`_WHEN_WIDTH` 2→4.
`_when_extent` DELETED (write-only; duration left the band).
ActiveEncoding's `[-5]` confirmed vestigial (activation is
modality-sliced, never tail-resolved) — docstring says so. Loss seams:
no code change needed — `_reverse_event_loss`/`_masked_event_loss`
already read the LIVE band from the subspace (now (2,4)); ModelLoss
constructor stays canonical_shape("OutputSpace"). Slot sweep: no other
literal tail offsets in bin/ (grep `[-4/[-3/[-2/[-1`, `..., -N`,
`n_when=2`); v1 `WhenRangeEncoding` class retained (self-contained, no
live constructors).

**Config-geometry ledger (the +2 when-band shift; nWhat preserved where
the dims encoded content+band):** model.xml IS/PS/Modal/CS nDim 5→7 (+
comments); MM_20M_legacy IS 5→7 + PS nInputDim 5→7; MM_400M 5→7 ×2;
MM_symbolic_iter 5→7 ×2; XOR_recon 6→8 (all sites); HeadEmission 12→14
×4; MM_5M_AR / MM_5M_IR / LM_5M_IR 12→14 + 1028→1030. OutputSpace
`nDim=4` sites untouched (band (0,0)). Configs that pin large interior
nDim (e.g. MM_20M family, 1024) keep their nDim — content narrows
1020→1018 there (absorbed in the re-baseline; no crash class).

**3.3 THE COORDINATED RE-BASELINE (old → new; epochs 3, seed 0,
cpu/eager, bin/recon_bench.py):**

| config | output_loss | recon_loss | exact | where_rec |
|---|---|---|---|---|
| `MM_20M_xor` | 0.186838 → 0.197179 | 0.015737 → 0.017832 | 0.5 → **0.75** | 1.0 |
| `MM_20M_legacy` | 0.124488 → 0.148598 | 0.073159 → 0.074824 | 0.0 | 0.0 |
| `MM_20M_grammar` | 0.175000 → 0.175000 | 0.002443 → 0.002389 | 0.0 | 0.25 |
| `idempotent` | 0.174948 → 0.174845 | 0.0 → **0.230116** | 0.0 | 0.0 |

(idempotent's reconstruction channel came ALIVE — the wider default
event arms the masked path on the inherited defaults; finite, its pin
asserts finite-only. Full-precision values in the `output/recon_*.json`
records, timestamps 2026-07-04 1049xx.)

**Pins updated (each flagged in its docstring):**
1. `test_mm20m_xor_roundtrip_at_harness_budget`: exact 0.5 → 0.75
   (measured deterministic E=3 point; where_recovery 1.0 kept).
2. `EPOCHS_PINNED` 38 → 80. Measured trajectory E={3, 10, 20, 30, 38,
   50, 60, 70, 75, 80, 90, 100} → exact {.75, 0, 0, 0, 0, 0, .75, .75,
   1.0, 1.0, .75, .75}; where_recovery 1.0 from E=50 up (dips 0.33 at
   10–38). **THE STANDING FORMULA'S STABLE-PLATEAU PREMISE DOES NOT
   HOLD**: 1.0/1.0 verified on [75, 80] ONLY; E=90/100 regress to 0.75
   (ONE row of the known content-association drift — decoded
   `'hello world' → 'hello hello'` class; tiling stays perfect).
   Isolation probes: NOT the `.where` period (old-period probe at E=30
   still 0.0), NOT the content narrowing (nDim 1026 probe still 0.0),
   NOT the when loss term (whenScale=0 probe unchanged) — the moved
   landscape is the representation change itself. Pinned at the
   verified 80; the [90, 100] instability is an OPEN finding for
   Alec's review at Gate A.
3. `test_grammar_fixtures` tense fixtures + `test_model_time_when`
   TenseLayer test ADAPTED to the v2 contract (4-wide tail, onset
   decode, one-seam encoder; extent assertions keep shape with a
   constant 0 — the band no longer carries duration).

**3.4 Serialized-model incompatibility:** verified empirically — a
checkpoint with old-band content widths (8→10 widened fixture tensors)
RAISES `ValueError` with the actionable shape-mismatch table under
`require_match=True` (the autoload migration-cliff catcher); non-autoload
soft-warns with the same table. Never a silent misread.

**Sweeps (all green):** roundtrip 19+1skip (THE bar 1.0/1.0 at E=80
under RUN_SLOW, 68.9s), recon_bench 4, radix_recon_render, handoff 7,
where-path 17, when/tense 26 (2 files adapted), when-v2 codec 11,
where-period 5, sparse_concept_e2e+cs_to_ss+subsymbolic_stack 32,
iterated_wave+symbolic_iteration+config_matrix+cs_sparse+sparse_layer
55+2skip+1xfail. Byte-lexer crash fixed by the config ledger
(MM_20M_legacy was building a 0-width IS what slab).

### Task 4 — blind decode + curriculum (Gate B; mechanism GREEN, bar RED)

**Built.** `PartSpace.decode_blind_rate` (None/0.0 scaffold — the 5c/5d
pin path, byte-identical; 1.0 fully blind; $0 < r < 1$ the
scaffold-masking curriculum, deterministic index stride so pins
reproduce; the free-run training consumer sets it to the XML maskRate).
Blind tiling in `_decode_radix_meta`: render set from the BAND alone —
per-slot magnitude MAX-GAP split (`_blind_active_slots`, ratio guard
1.8, absolute-floor 0.25 fallback; measured at the trained regime: real
slots 0.997–1.002 vs pads $\le 0.11$ — the split is clean, the fixed
"real stamp ~1.0" floor of the design would ALSO work there but fails
at early epochs where transport shrinks stamps to ~0.35); tile sizes =
next-active-claim minus the RUNNING SUM (the design's snap — emitted
bytes re-anchor cum each step, claim errors do not accumulate); arm
(a)/(b) association unchanged. recon_bench: `blind=True` DEFAULT,
`--scaffold` the explicit debug/regression mode, `decode_mode` note in
records; the roundtrip scaffold pins now pass `blind=False` explicitly
(flagged in their docstrings). Tests: `test/test_blind_decode.py` — 5
mechanism tests green (synthetic byte-exact stamps drive render-set /
size-hypothesis / curriculum / flag assertions; the default-is-scaffold
identity pin), + the RUN_SLOW bar.

**RESOLVED — THE BAR IS GREEN (2026-07-09).** The RED diagnosis below mis-framed
the cause as full-model band *precision* under a fixed period. The real cause was
`.where` **FREQUENCY**: at the 8192 default period 1 byte = 0.00077 rad, far below
the ~0.02 rad reverse-transport noise, so the reverse band collapsed. The
`[8,27,40]` "claim error" was mostly wrap-aliasing at the angle-0 seam (offset 0
wrapping to ~maxVal), not irreducible precision loss. Fix (3 parts): (1)
`WhereEncoding.where_origin` — a small (1/64-period) origin shift off the wrap
seam; (2) `<wherePeriod>256</wherePeriod>` on MM_20M_xor — the ~12-byte buffer
only needs a short period, so the higher frequency lifts the byte-offset signal
above the noise → the band decodes EXACT starts (start-offset error 0.0); (3)
NUL-exclusion in `RadixLayer.associate_span` for the content-terminated last
tile. Net at E=80: content 12/12, exact_match 1.0. `test_mm20m_xor_blind_roundtrip`
xfail removed (a real passing test). The training-pressure knobs (a)/(b) below
turned out NOT to be the lever (the band was collapsed by frequency, not
under-trained); (c) scaffold-fed was not needed. FOLLOW-UP: long-sentence configs
need a coarse+fine multi-rung period, not this single short period.

**[SUPERSEDED — historical] THE BAR IS RED — measured, diagnosed, escalated.**
`test_mm20m_xor_blind_roundtrip` (scaffold OFF, E=80): exact 0.0, where
0.33; with the snap, E=200: exact 0.0, where 0.17. Root cause is NOT
the mechanism (synthetic-stamp tests are exact end-to-end): the
FULL-MODEL band precision lags the fixture probes the E$\approx$80
crossing came from — decoded claims vs true $[0, 5, 6]$: $[8, 27, 40]$
at E=80, $[7, 5, 10]$ at E=200 (4–7 byte error, converging slowly);
the xor size inventory $\{1, 5, 6\}$ needs SUB-HALF-BYTE claims to
separate 5 from 6 (the design's own fixture floor was 0.73 bytes WORST
slot — not reached by the full-model transport at practical budgets).
Slot-activity gating is SOLVED (the magnitude split is exact at E=80).
OPEN — Alec's knob at Gate B: (a) more where-band training pressure
(whereScale / a dedicated band term), (b) longer/faster budgets, or
(c) accept scaffold-fed as the bar while the masking curriculum
(rate = maskRate) matures the band in free-run training. The bar stays
RED under RUN_SLOW per the nWhere-fix precedent; the default suite is
unaffected (`make test` does not set RUN_SLOW).

### Gate-B RED bar characterized (Alec's ask, 2026-07-04; scratchpad probes
`probe_blind_characterize{,2}.py`, `probe_blind_nul_exclusion.py`, E=80)

The blind failure decomposes into THREE measured defects; the activity
gate and running-sum placement are healthy.

1. **Band attractor collapse (the inverse derivation).** The reverse
   band is a DETERMINISTIC PER-SLOT CONSTANT, not the input's stamp:
   claims [8, 27, 40] identical across all four rows AND across
   consecutive evals (per-slot bias +8/+21.5/+33.5 bytes, spread
   $\le 1$). Unrounded angles show only $\approx 6\%$ input-conditional
   transmission (the 'hello'(5) vs 'loving'(6) tiling difference of 1
   byte appears as 0.06 bytes, below the 0.09-byte content-leakage
   noise) — the two tilings are NOT distinguishable from the band at
   this training state. Calibration can remove the bias but cannot
   restore per-input signal that is not transported; the fixture's
   0.73-byte floor came from direct codec-path training, which the
   full model's lossRev pressure does not reproduce (MSE against the
   batch-average stamp IS the attractor). The scaffold-masking
   curriculum (rate = maskRate free-run training) is the designed
   pressure against exactly this.
2. **NUL shadow (tokens after the derivation).** At every last tile,
   the NUL percept outranks the true word by 0.015–0.026 cosine
   (`'\x00'` 0.581 vs `world` 0.566 …); the true word is rank 1 of 18.
   Excluding NUL from arm (b) for gate-ACTIVE slots (active $\ne$
   $\emptyset$ by definition — the serial plan's recognize-NULL-on-
   decode principle) picks the true word 4/4.
3. **Cross-size content confusion (tokens, second instance).** With
   sizes unavailable, the 'loving' rows' FIRST tile associates to
   `hello` (unrestricted cosine ranks the wrong 5-byte word over the
   true 6-byte one). Scaffold mode never surfaced this: arm (a)'s
   size-6 bucket is the singleton {loving}. Transport health: content
   cos(rev, proto) 0.36–0.57, norms ×0.24–0.45 — content survives as
   RANK ORDER within size buckets, not across them; the size channel
   carries discriminative load, not just placement.

**Ablations (blind gating + placement throughout, E=80):** true sizes
ALL → **4/4 exact** (the bar failure is 100% the size channel); true
sizes except last → 0/4 (NUL shadow); NUL-exclusion alone on the full
blind path → **2/4** ('hello' rows exact incl. spans; 'loving' rows hit
defect 3); slot-0-anchored calibration → 0/4 (the bias is not a pure
translation). Store size inventory {1, 5, 6}.

**Implications (Alec's knobs, not applied):** (a) the curriculum
training run is the designed fix for defect 1 — the bar should be
re-measured after a rate=maskRate training phase; (b) NUL-exclusion for
active slots is principled and necessary (rescues half today); (c)
defect 3 wants either band-supplied sizes (via (a)) or a cross-size
association rule (e.g. maximality preference / per-bucket best). The
Gate-A E=90/100 scaffold instability is the same association fragility
surfacing under a different guise.

### Task 5 — docs + Gate C (2026-07-04)

Docs updated: Architecture.md sec D (period claim corrected — config-derived
`<wherePeriod>`, was wrongly "½·InputSpace"; blind-decode pointer) + the
contiguity-read note (`.when` left the bracket); Spaces.md encoding section
(the full 4-dim `.when` ladder shape + shared-HF caveat + clock side-band +
the config-derived `.where` period) + the retired-knobs note ((2,4) band);
Params.md (the three new `<architecture>` knobs + the band table rows);
Mereology.md + the Architecture three-aspect-loop note (`.when` no longer
bracketed). `make doc` NOT run (pandoc absent on this host — pre-existing;
the markdown is the source of truth).

**GATE C — `make test` GREEN: 2996 passed / 0 failed / 49 skipped / 32
xfailed / 7 xpassed, 641s serial (the canonical deterministic gate).** The
15 band-fixture failures from the first full run were all adapted (canonical
(2,2)→(2,4) pins; `.when` bracket→ladder decode helpers in
test_event_compose / test_tense_aspect / test_grammar_fixtures /
test_model_time_when / test_basicmodel; the +2 config-nDim ledger reaching
test_config_scoping + test_perceptualspace_bpe_forward inline configs;
the WhenRangeEncoding→WhenStartDurationEncoding type pin).

**Suite optimization (Alec's ask, 2026-07-04):** the serial suite is ~11
min. Added pytest-xdist (requirements.txt) + an OPT-IN parallel path:
`TEST_JOBS=auto` in `test/test_report.py` (`-n auto --dist=loadfile`, one
file per worker) and a `make testp` target. Measured: **2997 passed / 0
failed in 175s (2m55s) — ~3.6× the serial 641s**, identical green (the ±1
pass/skip is collection-order variance, zero failures both ways). Default
`make test` stays SERIAL + deterministic (a few tests share fixed `output/`
checkpoint paths; loadfile keeps same-file tests on one worker but the
canonical gate stays serial to be safe). The model's module singletons
(TheXMLConfig/TheData/TheGrammar) are per-process, so xdist workers are
naturally isolated.

## Self-review (writer, 2026-07-04)

- Spec coverage: design sec 1 → Task 1+4; sec 2 (decided shape) → Task 2;
  sec 3 plumbing/re-baseline/docs → Tasks 3+5; addendum micro-probe →
  Task 0.2; the four review defaults are declared in the header.
- No placeholders: codec formulas, slot shifts, and test intents are
  concrete; evidence-contingent numbers (budgets, re-baselines) are
  explicitly measured-at-execution with the standing formulas named.
- Type consistency: `WhenStartDurationEncoding` (Task 2) is what Task 3
  plumbs (nWhen=4) and Task 5 documents; `<wherePeriod>`/
  `<whenDurationPeriod>` names used consistently; `debug_scaffold` flag
  named identically in Tasks 4.1/4.2/4.3.
