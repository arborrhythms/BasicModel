# B — seed-23 cause

The recorded-source replay reproduces every archived per-batch R, final R
1.2357598543167114, and compose chooser delta **0.0**. Its instrumentation is
in `seed23-recorded-source/measurement.json`. It starts a fresh model and runs
exactly the original six batches; it does not append updates to the failed run.

The cause is **no non-tied compose exposure**, not a zero proposal scale. Each
batch contains two independently selected departure owners. The old receipt
reported their mean R next to only the compose reducer's parameter delta.
Those aggregates incorrectly suggested that the compose chooser owned each
unequal pair. In all three sampled compose departures, that row's R+A ties.
The non-tied rows belong to the candidate-attention owner. Exact ties correctly
have no policy graph and no momentum-only update.

| Training batch | Departure owners, rows 0 / 1 | Compose selected scale | Compose row Δ(R+A) | Compose loss / gradient |
|---|---|---:|---:|---|
| 0 | candidate / narrowing | — | — | 0 / 0 |
| 1 | narrowing / candidate | — | — | 0 / 0 |
| 2 | narrowing / compose | 51 | 0 | 0 / 0 |
| 3 | compose / candidate | 51 | 0 | 0 / 0 |
| 4 | compose / candidate | 60 | 0 | 0 / 0 |
| 5 | narrowing / candidate | — | — | 0 / 0 |

The initial replays of the reviewed working tree diverged at batch 4 because
the archived measurement predates two source edits at 17:49 UTC: the prime
changed from `for − against` to `for + against`, and inactive STM rows retained
their source lanes. Restoring those recorded functions in an isolated diagnostic
process reproduces the archive. Replaying all four arms in their original order
and reverting retention normalization did not account for the divergence.
The diagnostic reconstruction exists only in its receipt script; production
keeps the correct independent evidence lanes and has no alternate implementation.

Repair: preserve exact-tie disconnection and positive proposal correction;
assert a positive finite correction for every credited departure; report the
owner, its non-tied exposures, its actual gradients and parameter changes
separately. The production lane/prime fix already present in the reviewed tree
produces compose exposure in this diagnostic (delta 70.33487840276212), but this
is **not a regrade or learning pass**. The old execution/update and learning
gates remain false. A new repaired-tree execution receipt must use the declared
protocol and all three owners; an unexposed owner is reported as unexposed,
never as verified learning. Item 0 and 5.5 remain unchanged.
