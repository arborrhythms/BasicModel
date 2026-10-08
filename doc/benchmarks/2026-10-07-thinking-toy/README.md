# The thinking loop's credit: the fixed point before item 6.2 (Claude, 2026-10-07)

A chooser over thought actions (direct lookup, chain by transitivity,
conclude) with a work budget of four, credited exactly as compose is:
greedy episode against one uniform departure, surrogate `K·R·p(a_dep)·ΔC`.
Two credits: the answer (the open reference filled with the right thing)
and expectation alone (the next sentence states the consequence; reward is
its predictability). 20 seeds, 300 episodes each.

| conclude rule | answer credit | expectation credit | failure mode |
| --- | ---: | ---: | --- |
| conclude legal at any time (`sim_free_conclude.py`) | 14/20 | 9/20 | every failure concludes immediately; one departure toward a two-step answer looks worse than stopping, and longer training does not help (1,500 episodes: same) |
| conclude legal only once the open reference is filled (`sim_open_reference.py`) | **20/20** | **17/20** | — |

Alec's definition — a question is a row with an open reference — removes the
trap structurally: while a reference is open and work remains, concluding
is not a choice; thought must act on the reference, and the chooser learns
which actions fill it. Expectation alone teaches chaining in most seeds
without a supplied answer.

## Can a VP learn `plus` over opaque numerals? (`sim_plus_vp.py`, 2026-10-07)

Numerals 0–19 as opaque identities; meanings as context means over the
sentence codes of the facts each numeral appears in (counting facts plus the
training addition facts); a verb map (an MLP over the two operands'
meanings, read out as the nearest numeral) trained on 70% of the pairs and
tested on the held-out 30%; three seeds.

| meanings | train | held-out pairs | chance |
| --- | ---: | ---: | ---: |
| from co-occurrence (random sentence codes) | .33–.41 | **.00** | .05 |
| a smooth number line (what rich ordering contexts would give) | .31–.39 | .13–.27 | .05 |

Over opaque identities whose meanings are random co-occurrence sketches,
the map memorizes seen pairs and generalizes to none — there is no
structure to carry. With a number line in meaning it generalizes a little,
above chance but far from exact. Addition that generalizes needs thinking:
the successor facts applied in sequence, with the worked steps in the
corpus as the signal that teaches the chain.
