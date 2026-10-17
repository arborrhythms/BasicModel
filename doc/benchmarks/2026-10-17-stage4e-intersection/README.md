# Stage 4e: intersection and lesson-owned fold pressure

The reviewed orders/scorer preparation landed at BasicModel `89621bc61`,
WikiOracle `c78c292`, with no learning claim. This candidate is archived
as `6f78e9b25` on `codex/stage4e-intersection-pressure`.

Adjective descriptions now target `intersection`, as decided in the
[operator catalogue](../../specs/2026-09-29-operator-catalogue.md). No
`compound` target occurs, so this corpus needs no case lesson. Native
teacher traces complete without fabricated cases. The scalar fold-pressure
gain is a compose lesson parameter initialized at its previous configured
value; no other logit term changes. Compose lesson parameters use the order
lesson's Adam settings at the same .003 learning rate. The focused
[regression](gain-regression.txt) verifies ownership, matching optimizer
settings and wait CE reaching and changing the pressure gain.

## Pilot gate

The [pilot](pilot-pilot.json) restores the accepted paired stages-1–2
warm-up, seed 20261010: 16 training presentations, four batches, eight
paired optimizer steps, 167.69 seconds. The fixed probe is the first
annotated sentence, `a horse sees the cat`, captured before the first
update (the native callback is per row, not per batch). It contains five
wait states and four fold states; no extra presentations collect it.
The gate declared before execution requires both raw mean CEs to decrease
by more than 1e-5 nats.

| Fixed-state diagnostic | Before | After |
| --- | ---: | ---: |
| Wait CE (five targets) | 2.529912 | 1.715901 |
| Fold CE (four targets) | 3.299754 | 2.667019 |
| Fold-pressure gain | 1.000000 | 1.015367 |

**Pilot passed.** The one full rerun restores the original checkpoint and
seed; the pilot's ending state is discarded. Its fixed budget is 72
training sentences twice (144 presentations, 72 paired optimizer steps),
followed by 40 free held-out readings. An annotation never replaces the
student's kept program. No identity or 5.5 clock/chain/surface run follows
this pass. Full results and the final default sweep will be recorded here.
