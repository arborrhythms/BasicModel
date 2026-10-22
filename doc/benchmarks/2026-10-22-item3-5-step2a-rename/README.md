# Step 2a: class rename — 2026-10-22

Authority: [step-2 spec §2.0 and §§10–12](../../plans/2026-10-21-step2-one-ltm.md).
Base: R10 landing `e243d7bdf`, published with WikiOracle `2ffda78`.
This is 2a's separate rename commit before any column cull or layout change.

`Layers.TernaryTruthStore` becomes `Layers.LTM0`. `PartStore.PartStore`
becomes `LTM1.LTM1`, with the module moved to `bin/LTM1.py`. No old alias
or shim remains. Attributes (`ltm_store`, `parts`), mixins, ownership,
layouts, field names, state-dict paths and checkpoint formats are unchanged.
All references in code and tests, config comments and live contract docs
use the new names. Historical receipts retain their old source facts.

Before editing, a raw-byte and pickle-opcode audit inspected 91 saved files:
all available checkpoint files under output/data/test, the two binary test
fixtures, the five live replay inputs and two retained decomposition copies.
All eight phase registry entries are null. No audited file contains either
old class name anywhere; no class-path dependency is found. No model was
deserialized or rewritten, and retired artifacts were not read. Paths,
hashes and pickle globals are in [the audit](checkpoint-class-paths.json).
The checkpoint writer exports state dictionaries and allocator/table data,
not these module instances.

A byte comparison against the base checks that every changed code, config
and fixture file differs only by the two declared identifier substitutions.
The sole move is `bin/PartStore.py` to `bin/LTM1.py`. No assertion semantics
or body changes beyond class references; no new assertion, seed, grammar
rule, objective, detach, pool read or working vector.

Freeze this source, then run the unchanged eight-file set and regression
files on this hash. The pending-premise cases capture their initial state,
resolver choices and keep probes; all failures are retained. No earlier
result is promoted and no full sweep runs here. After these gates, 2a may
continue with only the safe culls; Claude reviews the round before its
publication. Step 2's one full sweep belongs at its end.
