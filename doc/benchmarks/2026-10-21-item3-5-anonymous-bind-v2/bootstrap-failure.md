# Bootstrap-sharing gate: retained failure on `823f157cd`

The eight-file set on `823f157cda4ef6b08744591ade7897476f8551b6` reports
112 passed, one failed and one existing copular prerequisite skip. The
failure is the unchanged `test_both_training_trials_share_bootstrap_snapshot`
at line 165: it requires every trial's reconstruction component to be zero.
In batch 2 the original run has R about .0070 for greedy rows 0 and 2,
and zero for explore. That initial draw did not have a bootstrap RNG probe;
its complete bounded-run log and source manifest remain in `native-final`.
It is not replaced by a later draw or called a pass.

`bootstrap-diagnostic-01` is a separate first diagnostic draw, captured from
initialization with process RNGs and all initial parameters. It fails at
the same assertion, with R = .00700668478384614 for both trials in rows
0 and 2. `bootstrap-replay-02` reproduces its credit exactly. This is a
read-only diagnostic, not another attempt to replace the native gate.

The capture was also run against an isolated archive of accepted source
`eac2fe87b2e2f0d24ee16fe8250dec37e59abbc7`, in `bootstrap-baseline-03`.
Initial parameters and every captured batch are identical across the two
sources: readings, programs, primed bank, reconstructed vectors, named
costs, R/E/A, keep decisions, pool and reference audit. Both fail.
[The comparison](bootstrap-source-comparison.json) records the equality.
No old-source pass is promoted.

| Batch 2, rows 0 and 2 | Greedy | Explore |
|---|---:|---:|
| R | .00700668478384614 | .00700668478384614 |
| Free-byte reconstruction | 0 | 0 |
| Forward-discard penalty | 0 | 0 |
| Relative content reconstruction, before .1 weight | .07006684690713882 | .07006684690713882 |
| Reference credit | absent | absent |
| MTM entries | 0 | 0 |

`bootstrap-bank-04` adds explicit bank tensors to the same replay. The
forward word targets equal their primed bank rows exactly. The decoder's
returned meaning for the first word in rows 0 and 2 has smaller magnitude
(for example .08606629818677902 instead of .09622504562139511); the form
still spells correctly. Both trials use the same bank and return the same
content discrepancy. `_component_reading` is false throughout. This
isolates the failed zero-R assertion from anonymous binding, but does not
waive the failure or establish that the decoder's discrepancy is intended.

The assertion, objective, keep rule, fixture and production source remain
unchanged. The failure needs disposition before this landing can claim a
green eight-file gate. The diagnostic observer `bootstrap_probe.py` was
added after the source freeze; it only saves already-computed state.
