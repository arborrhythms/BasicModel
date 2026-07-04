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
