# Native curriculum readout

**The fixed-budget measurement failed its learning gate.** All four arms ran
both native sentence trials and updated the shared predictor. The graded seed
23 chooser did not move, so the protocol's execution/update check also failed.
The process completed normally in 158 seconds, peaking at 1.20 GiB, within its
600-second / 8-GiB limits. No extra training was run after this result.

| Seed | Order | Held-out R before | Held-out R after | Chooser L1 parameter change |
|---|---|---:|---:|---:|
| 19 | graded | 1.266717 | 1.240000 | 34.279896 |
| 19 | shuffled | 1.266717 | 1.247497 | 36.734112 |
| 23 | graded | 1.245723 | 1.235760 | 0.000000 |
| 23 | shuffled | 1.245723 | 1.158409 | 29.485001 |

Graded order beat shuffled order for seed 19, but lost for seed 23. Lower R is
not a certificate of a correct derivation: the carried Item 0 zero-cost
counterexample remains relevant. This first native measurement records costs,
round counts and closing depths; it does **not** establish the spec's proposed
derivation-optimality rate, repeated-cat answering, collapse, or mature learning
gates. Those remain open. In particular, the seed 23 stall is retained evidence,
not excluded from the result or averaged away.

The [fixed protocol](native_protocol.md), [harness](native_experiment.py),
[full measurement](native_results.json), and [process receipt](native_process.json)
record the corpus, exact order, both seeds, all trials and source fingerprint.
Subsequent retention metadata fixes have their own regression checks; this
receipt is evidence for the recorded source, not a claim about a later rerun.
The prior Item 0 findings and independent 5.5 budgets and gates are unchanged.
