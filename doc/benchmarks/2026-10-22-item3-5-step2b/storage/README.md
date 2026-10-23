# Storage-only 2b: partial work and the relative-occurrence port stop

2026-10-22. Input source on main: `bb0045d7a`, which commits Claude's §38
edit unchanged (SHA-256
`393975a54b7519a0797a7979e938c4e3571bc7d801db055caf0dd31bc59c5fa1`).
The authorized scope is §36, with P5 amended by §§37–38.

**Not frozen and not gate evidence.** Implementation stops at
[spec §40](../../../plans/2026-10-21-step2-one-ltm.md#40-relative-occurrences-and-the-operand-only-certificate--codex-2026-10-22-for-claude-and-alec).
The additional relative sentence carries the outer close's provenance;
two existing checks in `test_relation_closing` previously covered only
its unasserted operand rows. The [exact proposed port](proposed-relative-occurrence-port.patch)
preserves their operand obligation and separately checks the sentence.
It is unapplied and requires Claude's review and Alec's decision.

The working tree contains partial implementation of the three removals,
LTM0 `refs` retirement, separate occurrence/edge evidence, relative
occurrences and staged forgetting. Forgetting has a draft pass-wide
dependency snapshot and a scalar unfinished-pressure flag so a staged
pass can continue toward low water at a later boundary. These edits have
not had their final static audit or any runtime validation. MTM,
ReferenceClose, PartClosing, PartRead and ClauseJournal remain unchanged.

Applied test work so far:

- P1's retained question kind/evidence check.
- P3's removal of only the centroid-switch conjunct; its form snapshot
  assertion is unchanged.
- P5's approved two complete-model pair-corruption cases and the
  no-saved-postings assertion in `test_part_read`.
- Initial LTM0 `refs` absence/setup ports and the authorized stale
  `where` removal in the cross-path comparison.
- The three authorized tests that check only adjacency or its nudge are
  retired: `test_centroid_moves_without_losing_identity_and_caps_containment`,
  `test_adjacent_wholes_rewitness_compact_and_checkpoint` (both in
  `test_operators_final.py`) and
  `test_absent_adjacent_wholes_leave_the_lower_form`
  (`test_review14_contracts.py`).

Exact records:

- [Input file hashes](input-audit.json).
- [Working production, configuration and test diff](working-source.patch),
  including every fixture edit and retired test body.
- [Applied assertion changes](applied-assertion-diff.json), with exact
  before/after text and source locations, including exception assertions.
- [Stop audit](stop-audit.json), with hashes of all 29 changed source/test/
  configuration files and the unchanged deferred modules.

Remaining work includes the other authorized absence and restore ports,
the two evidence tuples, exact completed-S counts, the two-pass coarsening
fixture, new scoped regressions, the final source audit and freeze, and
all gates. No test, diagnostic, gate, failed test run, publication or
WikiOracle bump occurred. Static source reads, AST extraction and patch
construction do not count as gates. Earlier receipts remain unchanged.

After the port decision, resume this working tree on main. The required
gates remain the regression files, the eight-file set and files covering
every changed module, all on one frozen hash, followed by Claude's review.
The identity line needs Claude's spec before it starts. Item 3.0 has not
been reached.

During this preflight, a concurrent [§39 amendment](incoming-rule-amendment.md)
recorded permission to document fixes and continue to end-of-round review.
The chat handoff still specifies a stop, so confirmation of that superseding
rule has been requested in the chat. No reply has arrived at this snapshot.
The incoming amendment is preserved verbatim; the test-port review is §40.
