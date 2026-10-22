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

Frozen source: **`eb6980e7de493e8b35791404851ef749acee6a08`**.
The [source audit](source-audit.json) checks all 90 changed code, test and
config/fixture files against the exact identifier substitutions. The
[source diff](source-diff.patch) records the rename and module move.

The one requested run on this hash passes:

| Gate | Passed | Existing prerequisite skips | Result |
| --- | ---: | ---: | --- |
| Eight-file set | 113 | 7 | [native/result.json](native/result.json) |
| Regression files | 85 | 0 | [regressions/result.json](regressions/result.json) |
| Total | **198** | **7** | [gate-summary.json](gate-summary.json) |

The seven skips are the already registered copular identity (one),
re-identification (four), and live pending-premise learning (two) gates.
The forced pending-premise cases capture their initial state, resolver
choices and keep probes: forward **0 bound / 8 open**, reverse **0 bound /
16 open**, with zero attention budget and no closure forcing. This is the
eight-file gate's two first draws, not a fresh twelve-draw series. No
earlier result is promoted and no full sweep runs here. There are no
failed gates on this source. Step 2's one full sweep belongs at its end.

Before culling, code inspection found that several examples in §10 do not
yet have the asserted derived source. The dated [spec §14](../../plans/2026-10-21-step2-one-ltm.md#14-codex-cull-preflight--2026-10-22-for-claude-and-alec)
lists each reader and existing assertion, and recommends retaining the
fields until their replacements and any exact representation ports are
decided. No cull or port was implemented. `cull-preflight.json` is a
read-only inspection of isolated rows, not another gate. Its first script
and `cull-preflight.log` retain a setup rejection: a probe used the default
fact admission for an interrogative. The second invocation supplies the
intended unverified kind and records the information gaps; production and
tests stayed frozen throughout.

[Claude's completed review](claude-review.md), model `claude-opus-5-5`,
approves publication of frozen source `eb6980e7d`. Its requested additional
WikiOracle import search is clean; the command, exclusions and result are
retained in `parent-import-search.json`. The reviewer could not inspect
that parent from its restricted BasicModel working directory, so Codex
checked it separately. No additional test or collect-only pass was run.
The complete request, review stream, stderr and execution metadata are
retained.

Claude confirms that none of the proposed culls currently has a sufficient
derived source. Keeping those fields now follows §10; changing their
future disposition needs Alec. [Spec §15](../../plans/2026-10-21-step2-one-ltm.md#15-claudes-review-of-the-rename-and-culls--2026-10-22-decisions-for-alec)
records the reviewed choices and stops further storage changes. The
rename alone is ready for both mains and the WikiOracle bump. The rest of
2a and all of 2b/2c remain unfinished. No learning claim is made, and item
3.0 has not been reached.
