# Static review of the restated 2b layout — 2026-10-22

Requested work: review step-2 §31 and re-audit P4/P6/P7, then stop for
Claude and Alec. The review is [spec §32](../../../plans/2026-10-21-step2-one-ltm.md#32-codexs-review-of-31-against-the-code--2026-10-22-stopped-for-claude-and-alec).

- Main input: `e0d474090`, with Claude's uncommitted §§29–31.
- Production/test/config source: published 2a
  `0b030e3ebdeca87f9ebf80739237437c3c75ea8f`, unchanged.
- Incoming Claude edits: [exact patch](incoming-claude-sections-29-31.patch).
- [Source audit](source-audit.json): branch, HEAD, initial status, empty
  production diff and hashes of tracked production/test/config files.
- [Reader search](reader-search.txt), supplemented by the focused reads
  cited in §32. A match is not automatically a stored-derivation reader;
  live query/answer references have been distinguished in the review.
- [Proposed ports and fixture dispositions](proposed-ports.md), with
  [machine-readable exact assertions](assertion-ports.json).
- [Assertion inventory](assertion-inventory.json): source locations and
  original text from the audited files, including unchanged assertions.

No production, configuration or test file was changed. No test, diagnostic,
gate, freeze, delegated review, publication or WikiOracle bump was run.
File reads, AST extraction, hashing and document consistency checks are
static review work. One draft ledger-building command stopped on a
mistyped source line (`test_selected_nested_meaning.py:85`, a non-assertion
line); the corrected ledger uses the AST's exact source spans. This was
neither a test failure nor evidence about production.

The recommendations and proposed ports require Claude's answer and Alec's
decisions. They are not new permissions. P1/P2/P3/P5 stand as authorized;
P4/P6/P7's derivation-dependent proposals are superseded only if approved.
All other assertions stay. Additional existing-test or production failures
still stop the implementation round. The earlier sealed reviews remain
untouched. Item 3.0 has not been reached.
