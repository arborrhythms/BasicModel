# `.where` / `.when` Encoding Pass — Design (DRAFT for Alec review)

> **STATUS: DEFAULTS CONFIRMED (Alec, 2026-07-04) — the four plan-header
> defaults (hybrid clock addressing; config-derived `<wherePeriod>` = 8192;
> `<whenPeriod>` $10^6$ / `<whenRungRatio>` 32; blind bar BESIDE the
> scaffold bar) are confirmed. Execution starts on Alec's explicit go.**
>
> **SCAFFOLD-MASKING CURRICULUM (Alec, 2026-07-04):** the scaffold is a
> TRAINING BRIDGE, not just a fallback — when `scaffolding==true`, mask
> the EXISTING masking-rate fraction (the XML maskRate parameter, e.g.
> 0.15) of the scaffold tiles: the decode must recover masked tiles
> BLIND while the rest are given. This bridges scaffold-fed $\to$ blind
> as training allows ("training by dropping a part of the surface or
> scaffolding, as a bridge until complex ideas can be decoded"), and
> makes the blind bar's feedback DENSE (per-masked-tile signal). The
> plan's Task 4 (blind decode) implements the mask-fraction mode; full
> blind = mask rate 1.0. Successor to the
> reconstruction-fidelity plan's Gate-2b thread; every parameter below
> traces to a measurement in the precision probes (scratchpad
> `probe_where_precision.py` artifacts; pre-fix baselines in
> `prefix_where/`; findings recorded in the reconstruction plan's
> EXECUTION NOTES).

## Decisions already made (Alec)

1. **Blind recovery is the bar (2026-07-04):** reconstruction must
   re-derive the tiling from the representation alone; the 5c/5d forward
   scaffold demotes to a fallback/debug path. Gate 2b =
   exact round-trip with the scaffold OFF.
2. **`.where` stays 2-dim (2026-07-04):** with the nWhere loss wiring
   fixed, the existing quadrature pair measures 0.41 med / 0.73 max bytes
   on the Gate-2b regime at $P=8192$-equivalent — sufficient. The 3-dim
   fine-rung extension is NOT built. 8192 characters is accepted as the
   per-sentence cap; prior sentences are addressed by `.where` + `.when`
   jointly.
3. **`.when` needs real capacity (2026-07-04):** 2 dims cannot address
   LTM with per-position accuracy over $\sim 10^6$ events (arithmetic in
   sec 3). Decoupling start/duration (Alec's 4-value proposal) is the
   base improvement; the options differ in how start reaches $10^6$.
4. Standing: GPU-first device policy; fail-loud (silent-channel pattern
   now fixed at three sites; the fourth-instance guard applies here);
   Alec commits at gates.

## Measured facts the design builds on

- Codec is exact (float-rounding only) at every period tested.
- Post-loss-fix, sparse-support (fixture) training reaches the
  trained-claims floor $5.6\times 10^{-4}$ rad ($\approx 0.73$ bytes at
  $P=8192$, worst slot; deterministic per-slot bias, calibratable).
- Dense-support training floors at $\Delta\phi \approx 0.09$ rad
  regardless of loss wiring (carrier mixing in the forward+reverse
  transport through the shared event) — one quadrature pair reliably
  distinguishes $\approx 2\pi/0.09 \approx 70$ positions under dense
  support, $\approx 11{,}000$ under sparse support.
- Current `.where`/`.when` share one codec (`QuadratureEncoding`, one
  sin/cos pair, endpoint-sum bracket: angle = center, magnitude =
  extent). `maxVal` (the period) is $\Sigma$ nVectors $= 131{,}337$,
  raised-never-lowered at the build seam (bin/Spaces.py:9383-9391) —
  NOT the doc-claimed "$\tfrac{1}{2}$ InputSpace" (Architecture sec D is
  wrong and gets corrected in this pass).
- `.when` already stores EXACT time outside the band: "the long-int
  clock owns the EXACT time, the angle is the coarse/folded feature"
  (WhenRangeEncoding docstring) — the side-band precedent.

## 1. `.where` changes (small, mostly parameters + decode)

1. **Period decoupling:** the `.where` period becomes config-derived
   (default: the input/sentence cap, 8192; NOT $\Sigma$ nVectors). A
   16$\times$ resolution gain at zero dims. The build seam keeps
   raise-to-fit semantics for inputs larger than the configured period.
2. **Overflow warning (Alec's design):** if an input's byte length
   exceeds the configured period, WARN once (config name, length,
   period, "increase `<wherePeriod>`") — never silently alias. Site:
   the existing raise-never-lower seam.
3. **Blind decode (Gate 2b):** `_decode_radix_meta` derives the render
   set from the BAND (coarse position claims $\to$ tile hypotheses,
   snapped by running-sum consistency and percept byte-sizes), not from
   `_forward_input['tile_spans']`; the scaffold path remains as an
   explicit debug/fallback mode. Bar: `test_mm20m_xor_blind_roundtrip`
   — scaffold OFF, exact_match == 1.0 at the pinned budget (expected
   larger than 38 given the E$\approx$80 byte-exact crossing; measured
   at execution, pinned by the plan's formula).
4. NOT built (recorded): fine rungs for dense-corpus `.where`
   precision; revisit with a dense-corpus probe when corpus-scale
   training arrives.

## 2. `.when` capacity redesign (the substantive change)

**Requirement (Alec):** address LTM with per-position accuracy over
$\sim 10^6$ events.

**Base change (all options):** decouple start and duration — Alec's
4-value proposal. Start and duration each get their own quadrature
pair; the endpoint-sum bracket's angle/magnitude conflation retires for
`.when` (magnitude no longer carries duration; both quantities get full
phase precision and independent gradients).

**The capacity arithmetic** (dense support governs LTM):
one pair $\approx 70$ reliable positions; $10^6$ needs
$\lceil \log_{70} 10^6 \rceil = 3$ rungs marginally (343k at 3,
$2.4\times 10^7$ at 4).

### Option A — pure-band ladder
Start = 4 rung pairs (periods e.g. $10^6, 1.4\times10^4, 200, 3$) +
duration pair = **10 dims**. Fully continuous representation;
biggest event-width change; rungs need dense temporal support to stay
trained (true for a live LTM).

### Option B — Alec's 4-value alone
Start pair + duration pair = **4 dims**. Honest capacity:
$\sim$70–11k distinguishable positions depending on support density —
does NOT meet $10^6$ per-position alone; adequate only if addressing
tolerates neighborhood-scale ambiguity.

### Option C — hybrid: exact clock addresses, band disambiguates (RECOMMENDED)
Retrieval keys on the **long-int clock side-band** (already exact,
already serialized): per-position accuracy over $10^6$ events is
guaranteed by construction, zero phase-precision demands. The band
becomes Alec's 4-value (start pair + duration pair, 4 dims): a
similarity feature for content-addressable/approximate retrieval and
the trainable channel, needing only neighborhood-scale resolution
($\pm$70 within a retrieval bucket is ample). This is `.when`'s OWN
documented architecture, completed — and symmetric with what `.where`
does at Gate 2b (band recovers, integer truth available as fallback).
Cost: **4 dims** + the retrieval-path change to consult the clock.

**Recommendation: C.** A (10 dims) is the fallback if exact-clock
addressing is rejected for retrieval-semantics reasons; B alone is
under-capacity for the stated requirement.

### DECIDED band shape (Alec, FINAL 2026-07-04): 4 dims = pure start LF/HF ladder

**Duration inventory finding (code, 2026-07-04):** the `.when` duration
is WRITE-ONLY today — `whenEncoding.decode_span` has ZERO callers in
bin/; the "straddles now" present-tense rule exists only in the
docstring; executed tense machinery (`TenseLayer`/`_lift_when`/
`_lower_when`/`shift_time`) rotates the CENTER and never reads extent;
aspect is explicitly retired (bin/Language.py:3106 no-op,
"duration/aspect retired"). Nothing is lost by removing duration from
the band.

**The shape:** $[\sin(s\,\omega_{lf}), \cos(s\,\omega_{lf}),
\sin(s\,\omega_{hf}), \cos(s\,\omega_{hf})]$ — two TRUE quadrature
pairs, ONE quantity (onset), a 2-rung ladder. Constant norm (both pairs
exactly unit; nothing to calibrate, no validity keying, no sign
conventions); decode = two atan2 + branch resolution of HF by LF (safe
at rung ratios $\lesssim \pi/\Delta\phi \approx 35$ at the dense floor;
capacity $\approx 70^2 \approx 4900$ dense / $11k^2$ sparse
distinguishable onsets — the band is the SIMILARITY channel; ABSOLUTE
addressing rides the exact long-int clock per Option C). `shift_time`
generalizes: exact phase rotation per pair at its own $\omega$.
Endpoint-sum bracketing retires for `.when`. DURATION moves to the
EXACT side: the record store holds exact extents (as the clock holds
exact time), feeding the future aspect implementation when perfect/
progressive tenses are actually built. Superseded intermediate shapes
(3+1 shared-HF, cos/sin, sign-split sines) are recorded in the git
history of this doc only.

**Shared-HF tension + resolution:** absolute branch-resolution of a
sin-only start-fine needs $P_{HF} \gtrsim 4\times$ the coarse error
($\approx 57$k events at the dense floor) — but duration wants
$P_{HF} \approx 2\times$ max-typical-duration (short). DEFAULT = the
shape as stated (shared short HF): start-fine is then a LOCAL PHASE
FINGERPRINT (sharp neighbor discrimination for similarity retrieval,
absolute branch unresolved) while ABSOLUTE addressing rides the exact
long-int clock (the Option-C hybrid). Variant (a) — a third frequency
giving start-fine a branch-safe mid period — is the recorded
alternative if absolute band-only fine-start ever matters.

## 3. Shared plumbing (whichever option)

- Muxed event layout changes (where 2 + when 2 $\to$ where 2 + when 4
  under B/C; +6 more under A). `canonical_shape`, the stamp sites, the
  decode sites, and the band-aware loss seam (`_reverse_event_loss`,
  per-call `nWhere/nWhen`) all take the new widths. The fail-loud
  fourth-instance guard: any path that cannot know its widths warns
  once, never silently what-scales.
- **Global re-baseline:** the event layout shifts for every config —
  byte-identity is impossible BY DESIGN here. Procedure: capture
  old-baseline harness records for the canonical set (xor, legacy,
  grammar, idempotent) pre-change; re-capture post-change; the plan
  carries the old$\to$new table; every value-pin updated with each
  change flagged. THE bar (scaffold and blind variants) re-measured and
  re-pinned by the standing formula.
- Serialized models with the old event widths do not load — accepted
  (clean rebuild; no migration shim). Called out at the gate.
- Docs: Architecture sec D period claim corrected; Spaces.md encoding
  section; WhenRangeEncoding docstring reflects the decoupling; the
  reconstruction plan gains a pointer here.

## Addendum: joint-resolution analysis (Alec's question, 2026-07-04)

Can start/end pairs combine to beat either alone?

- **Same frequency, independent endpoints: no** — each recovers at the
  channel error $\Delta\phi$; the joint decode adds nothing, and naive
  differences carry $\sqrt{2}\,\Delta\phi$ — UNLESS the band errors are
  CORRELATED (common-mode carrier noise), in which case the DIFFERENCE
  channel (duration, relative offsets) cancels it and decodes finer than
  either absolute endpoint. Evidence for correlation exists (the
  deterministic per-slot bias at the trained floor). **Pre-implementation
  micro-probe:** stamp two pairs into one event, measure inter-band error
  correlation; if high, durations are near-free of the common noise.
- **Same quantity at two periods: yes** — the fine pair's resolution over
  the coarse pair's range (range-resolution product multiplies). Family:
  LADDER (ratio $< \pi/\Delta\phi$ — $\sim$35 at the dense floor,
  noise-robust; Option A's topology), CRT/residue (max range, brittle:
  one flipped residue = gross error), VERNIER (beat-extended range,
  capped at $\sim \pi/2\Delta\phi \approx 17$ here, CRT-brittle —
  dominated by the ladder under our noise).
- **4-dim allocations ranked** (interval $(s, e)$, two pairs):
  (1) start@coarse + start@fine — max addressing capacity, $70^2 \approx
  4900$ positions at the dense floor (still $\ll 10^6$: pure-band
  addressing needs Option A's 3-4 rungs); (2) center@coarse +
  duration@fine — Alec's 4-value refined; absolute resolution from the
  coarse pair, duration at full precision in its natural range, plus the
  correlation bonus if it holds; (3) start@P + end@P — weakest unless
  the correlation bonus is large.

## Open questions for the review

1. Option C confirmed as the `.when` shape (or A / other)?
2. `.where` period default: fixed 8192, or derived per-config from the
   input spec (with 8192 as the ceiling)?
3. Duration pair period for `.when` (max event duration — same $10^6$
   range as start, or shorter, e.g. $10^4$?).
4. Should the blind-decode bar replace the scaffold bar in the plan's
   Gate 2, or stand beside it as Gate 2b (scaffold test kept as the
   content-identity regression pin)?
