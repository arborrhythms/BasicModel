# Identity threading: mediated review hand-off

For Alec to pass to Claude. No direct review is requested by Codex for
this source. Publication of BasicModel main and the WikiOracle bump waits
for Alec's mediated review hand-off.

**Frozen source:** `65ab7fbe9d5c67271b0bea7e4853c8ba2b690363`, on main.
**Specification:** [identity threading](../../../../plans/2026-10-23-identity-threading.md),
§1 decisions and §32's final repair.
**Receipt:** [complete source/port ledger](../../README.md).
**Gates:** [source-specific summary](gate-summary.json): all 1,639 selected
cases completed, **1,627 passed, twelve existing skips, zero failures**,
132 distinct files and no compile-cache retry. No prior source's result
is promoted. The exact [source audit](source-audit.json),
[test ports](test-ports.patch) and [snapshot ledger](decomposition-snapshots.json)
are retained beside it.

The latest completed direct review, retained at
[the source-12 report](../source-12/claude-review.md), returned DO NOT LAND
for implicit typed operands' raw stored reference. Source-13 fixes that
correspondence in `ClauseJournal`, preserving the source-12 value lookup.
For identity programs, a known requested order other than one, excluding
kind/generic modes, may carry a different native operation reference to
the leaf. The code then exits the legacy branch: no formation, open or
mint decisions return. Order one stays owned by compose bind.

Eight new cases first failed on source-12, then passed unchanged after
the repair. They pass actual `prepare_operands` results into an identity
program and `finish_clause`, checking the main reference, factored
reference, composition leaf reference and exact typed value for
lift/part/whole/equal at orders zero/two. `typed-journal-01` retains every
pre-repair failure; `typed-journal-02` has 78 development passes. There
is no trial/keep in those deterministic unit calls. No existing assertion
changes and no test is retired in this repair.

The other review notes are accounted for: the literal exemption is
explicitly an identity-path choice, not claimed to match the ordinary
legacy path; the decomposition adapter directly checks current unary
names minus bind against the original sequence; and both event-history
notes remain linked in FutureWork. Original snapshots, seeds, parameter
rows, budgets, objective and keep rules are unchanged.

Please review the complete line and its exact `production.patch`,
`test-ports.patch`, `assertion-audit.json` and `source-audit.json`, with the
source-11/source-12 reports as context. Key boundaries remain MTM current
values and event histories, trial isolation, explicit compose-bind
authority, no writes from binds into concepts, no word as another word's
object, source/stream guards and unchanged pending/G1–G3/c/e contracts.
Today's derivation, readers, close-time minting and three-slot rows stay;
the future LTM0/LTM1 layout and shared STM chunks belong to item 3.0.

Return LAND or DO NOT LAND, with necessary repairs distinguished from
nonblocking follow-ups. The scoped pass is not a full sweep or learning
evidence; absent-artifact/platform skips remain skips. 2c follows this
identity landing, and item 3.0 has not been reached.

The final groups are 87 regression passes; 123 native passes/seven skips;
707 first-module passes/five skips; 706 second-module passes; and two
passes in each original-snapshot decomposition arm. The pending
certificate is forward 0 bound/16 open and reverse 0/32, with zero budget,
no closure forcing and no wrong binding. G1–G3 and c/e remain unchanged.
Nine skips need absent artifacts and three are opt-in slow MPS/Inductor
cases; no skip supplies learning or GPU evidence. The source, tests and
measurement fixtures are unchanged after the freeze.
