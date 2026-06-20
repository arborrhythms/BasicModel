# Model time and a serialized absolute clock for WhenEncoding

Date: 2026-06-07

Status: IMPLEMENTED 2026-06-07 (FINAL scheme: single scaled quadrature phasor;
angle = absolute time at period $65536$; magnitude $D \in [0, 1]$ = tense,
$0.5$ = present default; `next` / `previous` move $D$ by $\pm 0.1$; event
duration / aspect retired; the serialized `when_time` long int owns the exact
absolute time).

## Goal

Give the model an absolute, serialized notion of time and rebuild `.when` on
top of it:

1. A 0-initialized `long` "model time" that increments once per batch and
   serializes with the checkpoint.
2. `WhenEncoding` derives each event's `.when` from that absolute time as a
   single scaled quadrature phasor whose ANGLE is the (coarse) absolute time
   and whose MAGNITUDE is the TENSE position.
3. `next()` / `previous()` move the tense magnitude so verbs can express tense
   along this axis.

## Background (current state)

- The historical `TemporalEncoding` (pre-rename `bin/BasicModel.py` @
  `34e15c1`) declared `period = [1193, 2000147]` (two coprime primes) but its
  `forward` used `period[0]` for both dims (a copy-paste bug) and was never
  wired to a serialized clock. `PositionalEncoding` likewise declared
  `[65521, 65537]` and double-applied `div_term`. Intent: two coprime
  periods; implementation: broken.
- An intermediate `.when` design packed a signed interval into 2 dims via the
  endpoint sum $q(\text{start}) + q(\text{end})$ -- the ANGLE carrying the
  center time, the MAGNITUDE carrying the event DURATION
  ($2\cos((\text{end}-\text{start})\,\mathrm{dt}/2)$). That duration channel
  is RETIRED here (Section 2): the magnitude is repurposed for TENSE.
- `_training_step_count` ([bin/Models.py](../../bin/Models.py)) increments per
  TRAINING batch but is training-only and NOT serialized.

## Decision: magnitude is tense, not duration

With `nWhen = 2` the two real dimensions are a hard budget. The FINAL scheme
spends them on a single phasor: the ANGLE on absolute time and the MAGNITUDE on
TENSE. Event-extent duration is dropped (no separate duration channel); the
serialized `when_time` long int owns the exact absolute time, so the phasor's
angle only has to be a coarse local feature.

## Design

### 1. Model clock (`when_time`)

- `BasicModel` gains
  `register_buffer('when_time', torch.zeros((), dtype=torch.long))`,
  0-initialized. As a registered buffer it is part of `state_dict()` and
  therefore serializes through the existing `.ckpt` `save_weights` /
  `load_weights` path with no extra plumbing. Old checkpoints lacking the key
  load at 0 under the current tolerant (`strict=False`) path.
- Increment by 1 once per batch in `runBatch`, on BOTH train and inference
  (the clock advances whenever the model processes a batch).
- Accessor `present() -> int` returns `int(self.when_time)`.
- Each batch, `present()` is propagated to every live `WhenRangeEncoding`'s
  `self.t` (`BasicModel._advance_when_time`) so a default-stamped `.when`
  carries the absolute model time.

### 2. WhenEncoding: single scaled quadrature phasor

The `.when` 2-vector is

$$\texttt{.when} \;=\; D \cdot
\big[\sin(2\pi\, t / P),\; \cos(2\pi\, t / P)\big],
\qquad P = 65536,\; D \in [0, 1].$$

- The ANGLE encodes the ABSOLUTE model time $t$ (`self.t`, synced from
  `when_time`). The long-int clock owns the EXACT time; the angle is the coarse
  (folded / aliasing) feature, so it does not need a huge period to
  disambiguate distant epochs.
- The MAGNITUDE $D$ is the TENSE position: $D = 0$ past, $D = 0.5$ PRESENT
  (the default), $D = 1$ future. $D$ is tense, NOT event duration.
- Every component lies in $[-1, 1]$ (network-friendly); the magnitude lives in
  $[0, 1]$ (no magnitude-near-$2$ precision problem). At $t = 0$ the present
  default is $0.5 \cdot [0, 1] = [0, 0.5]$.
- `encode(t, D=0.5)` returns the scaled phasor above. `forward` stamps the
  present default `encode(self.t, D=0.5)` into the `.when` slots for every
  input event.
- `decode(when)` returns $(t, D)$ with
  $t = \operatorname{atan2}(\text{when}_0, \text{when}_1)\, P / (2\pi)$ (folded;
  the long int owns the exact time) and
  $D = \sqrt{\text{when}_0^2 + \text{when}_1^2}$ (the tense magnitude).

### 3. Period choice and float resolution

The period $P = 65536$ ($\approx 2^{16}$) is the single config knob (`maxT`,
defaulting to `_WHEN_PERIOD`; `_WHEN_MAXT` is a back-compat alias). The
per-time-tick angular step is $2\pi / P \approx 9.6 \times 10^{-5}$ rad; scaled
by the present magnitude $D \approx 0.5$ it is $\approx 4.8 \times 10^{-5}$ --
still $\gtrsim 400\times$ float32 epsilon ($\approx 1.2 \times 10^{-7}$), so
adjacent ticks stay distinguishable in the float32 `.when` dtype. The tense
steps are $0.1$ apart in magnitude (a huge margin). The serialized `when_time`
long int owns absolute identity, so $P$ governs only the LOCAL angular
resolution.

### 4. next() / previous()

- `next(k=1)`: $D \to \operatorname{clamp}(D + 0.1\,k,\, 0,\, 1)$ (toward
  future), preserving the time-angle; updates `self.D` and returns the
  re-encoded present phasor at the new $D$. `previous(k=1)`:
  $D \to \operatorname{clamp}(D - 0.1\,k,\, 0,\, 1)$ (toward past).
- On a `.when` tensor (`shift_tense(when, dD)`): decode $D$, set
  $D_\text{new} = \operatorname{clamp}(D + dD,\, 0,\, 1)$, and rescale the
  phasor preserving the angle, $\texttt{when} \cdot (D_\text{new} / D)$ when
  $D > 0$. When the old magnitude is $\approx 0$ (a zero phasor has no angle)
  the value is re-encoded from $(\texttt{self.t},\, D_\text{new})$. The
  divide-by-zero is guarded.
- A verb's tense stamps its event at present ($D = 0.5$), `previous`
  ($D - 0.1$, PAST), or `next` ($D + 0.1$, FUTURE) along the tense axis.

Edge: $D = 0$ (extreme past after five `previous()` from $0.5$) gives
$\texttt{.when} = [0, 0]$ (no angle); this is accepted -- the long int owns the
time -- and the rescale / decode guard the divide-by-zero.

### 5. Tense / aspect

- `TenseLayer`: PRESENT $\to D = 0.5$ (identity from the $0.5$ default), PAST
  $\to$ `previous` ($D - 0.1$), FUTURE $\to$ `next` ($D + 0.1$), operating on
  the `.when` MAGNITUDE (rescaling the phasor, preserving the time-angle).
  `_DELTA` is the tense step (PRESENT $0$, PAST $-0.1$, FUTURE $+0.1$) applied
  to $D$; `reverse` applies the inverse step. The round-trip is invertible
  WITHIN $[0, 1]$ -- clamping at the ends is lossy, so round-trip tests stay in
  the unclamped range ($0.5 \pm 0.1$).
- `AspectLayer` (SIMPLE / PERFECT / PROGRESSIVE): RETIRED to a no-op. Event
  duration is gone (the magnitude is tense now), so there is no duration
  channel for aspect to reshape; `forward` / `compose` / `reverse` / `generate`
  are identity. The class is KEPT (not deleted) so the grammar's `aspect` rule
  dispatch stays intact.

## Serialization

`when_time` is a registered buffer $\Rightarrow$ included in `state_dict()`
$\Rightarrow$ saved by `save_weights` and restored by `load_weights`
automatically. A round-trip test asserts `present()` survives save / load.

## Testing

- Clock: `when_time` starts at 0; `+1` per batch on train and inference;
  round-trips through save / load (`present() == N` after N batches); syncs to
  the live encoders' `self.t` each batch.
- Encoding: the present default stamp is $0.5 \cdot [\sin(2\pi\,t/P),
  \cos(2\pi\,t/P)]$ (magnitude $0.5$); at $t = 0$ that is $[0, 0.5]$; the angle
  tracks absolute `when_time` across batches; `decode` recovers $(t, D)$; every
  component stays in $[-1, 1]$.
- next / previous: $D$ moves by $\pm 0.1$ (clamped to $[0, 1]$) and the
  time-angle is preserved; `TenseLayer` PRESENT ($0.5$) / PAST ($0.4$) / FUTURE
  ($0.6$) compose, and PAST / FUTURE reverse round-trip in the unclamped range;
  `AspectLayer` is identity (no-op).
- Period: per-tick angular separation at $P = 65536$, scaled by $D \approx 0.5$,
  stays above the float32-eps resolution threshold (guards local sequencing).

## Non-goals

- No `nWhen` widening (stays 2; the two dims are the single tense-scaled
  time-phasor).
- No two-coprime center (ruled out by the single-phasor budget).
- Event-extent duration / aspect intervals are retired, not preserved.
- The `.where` two-prime regression (`[65521, 65537]`) is out of scope here.

## Resolved

The period $P$ (Section 3) is **65536** ($\approx 2^{16}$) -- the tune-for-local
choice: crisp local steps near the present, with the serialized `when_time`
long int owning absolute identity. The magnitude is the TENSE position
$D \in [0, 1]$ ($0.5$ = present default; `next` / `previous` move it by
$\pm 0.1$); event duration / aspect are retired (`AspectLayer` is a no-op).
Implemented 2026-06-07.
