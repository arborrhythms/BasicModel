# Storage-only 2b: P5 port conflict — 2026-10-22

Claude's four incoming documentation files were committed unchanged on
main at `cc6407f3b`. The preceding §32 static record was then retained at
`1db6e011d`. The current implementation scope is step-2 §36.

[Spec §37](../../../plans/2026-10-21-step2-one-ltm.md#37-p5-after-the-identity-first-split--codex-2026-10-22-for-claude-and-alec)
records a static conflict: the authorized P5 corruption replacement
tests `ref_kinds`, but §36 defers that column. The original corruption
case requires the saved `posting_codes` key that this round removes.

The [proposed patch](proposed-p5-existing-pair.patch) is **not applied**.
It replaces only that corruption subcase with two existing LTM1 `refs`
corruptions: a zero second argument on an occupied row, and a pair array
with only one column. The exact exception expectations, fixture changes
and original location are in the [source audit](source-audit.json).
Every earlier assertion stays verbatim; no test is retired. This is a
proposal for Claude and Alec, not an authorized test port.

Checks performed: source reads, patch syntax/AST inspection, source hashes
and comparison to published 2a `0b030e3eb`. The proposal parses and changes
no existing `assert` statement; its `pytest.raises` replacement is exactly
the additional authorization needed. No production, test or configuration
file was edited. No diagnostic, test, gate, source freeze, publication or
WikiOracle bump was run. There are no failed test runs in this preflight.

The earlier sealed records are unchanged. P1/P3 and P5's other clauses
remain authorized; tags, READING and P2/P4-tagged/P6/P7 stay deferred.
Stop here under the handoff's assertion-change rule. The identity line
still needs Claude's specification, the full sweep remains at step 2's
end, and item 3.0 has not been reached.
