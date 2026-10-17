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
this pass. The full run records documentation commit `d2adba780`; its
production source is unchanged from the pilot's `6f78e9b25`.

## Full rerun: gate failed

The [retained results](results.json) complete **144 training presentations
and 40 held-out readings**, seed 20261010, in **2,591.10 seconds** (43.19
minutes). The 72 paired updates ran as declared. The step wrapper counted
**73 optimizer calls**: batch 16 also took the existing native batch-end
path (`Models.py`, the `totalLoss.requires_grad` branch after sentence
updates). That path is separate from the two paired updates; no text
presentation or lesson was added. Per-call parameter changes were not
logged, so the additional call is not asserted to be a nonzero update.

All **288/288** native teacher calls have available targets (zero missing
cases). Raw mean CE across each epoch falls from **1.587939 to 0.911009**
for waits, and **2.496067 to 1.869878** for folds. These epoch means use
the same corpus, but evolving teacher states; they are separate from the
fixed-state pilot comparison. [Diagnostics](analysis.json) retain the
counts, kept programs, description stamps and closed meaning norms.

| Held-out group | Exact brackets | One correct NP occurrence | Complete order-1 lowering certificate | Unexpected occurrences |
| --- | ---: | ---: | ---: | ---: |
| Adjective NP | 0/8 | 0/8 | 0/8 | 8 |
| Intransitive | 0/8 | 0/8 | 0/8 | 8 |
| Transitive | 0/8 | 5/16 | 0/16 | 3 |
| Prepositional | 0/8 | 5/16 | 0/16 | 3 |
| **Main total** | **0/32** | **10/48** | **0/48** | **22** |
| Required | ≥90% | ≥95%, zero duplicates/collisions | Every annotated NP | 0 |
| Unlowered description controls | 8/8 | No occurrence: **8/8** | — | 0 |

**Zero duplicate writes and zero address collisions.** The description
controls pass, but the other three gates fail. Missing NPs remain in the
denominator; the description stamp is reported, not required to be 2.
The 14 matched NP-span lowerings all read description stamp 0. No NP
simultaneously has its required lowering and its own retained occurrence.
The ten matching occurrence descriptions are insufficient for that joint
certificate. There is one occurrence per main sentence, often on the wrong
constituent, rather than one per annotated singular NP.

The [kept programs](kept-programs.json) expose both order and folding
errors. For example, `a large cat` reads source positions `[0, 2, 1]`,
applies `lower` to `a cat`, then `whole` to the result and `large`.
Its occurrence describes only `cat`; the annotated description is
`large cat`. Only 9/32 main readings preserve source order, and even
those do not reproduce the brackets. This is an observed failure pattern,
not a causal ablation. Of 40 closes, 22 have a nonzero fused meaning point;
18 remain relative without one (`null`, not a zero vector).

Compared with the first run, main exact brackets worsen from 8/32 to
0/32; matching NP occurrences rise from 0/48 to 10/48 but the extra
occurrences also fail the gate. **CE movement is not a passing constituent
stage.** Stop here: no identity run, no further 5.5 clock/chain/surface
change, and no learned rule retirement.

## Default sweep

The one full [default sweep](sweep.json) on `d2adba780` completes all
5,800 cases in 483.35 seconds: **5,466 passed, 333 skipped, one XFAIL,
zero failures**. Relative to the accepted preparation's `6f76c081c`
sweep, the only changed existing outcome is c
(`test_forced_ordinary_answer_fills_committed_question_without_its_own_episode`),
which passes this time. This does not retire the broader structure finding.
The exact NOT/OR and paired zero-cost certificates pass unchanged; the
historical paired counterexample remains carried. There are no new failures.
Slow contracts retain their weekly tier. The stage-4e artifact gate will be
reported from the full rerun below, separately from this mechanism sweep.
