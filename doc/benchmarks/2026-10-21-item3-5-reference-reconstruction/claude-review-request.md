Review item 3.5 line 1 for publication, read-only. Do not edit files or run
another gate sweep. The user explicitly requires Claude's review before
both mains are published.

Read `doc/plans/2026-10-21-reconstruction-reads-the-bind.md`, especially
§§8–13: those contain your prior review answers and Alec's decisions. The
frozen source is `3a9db635e69e87c17dc50c13c42c2f3eee10910f`, built on the
grammar contract commits `15f154dca` and `eb858be39`, from `bb1678d17`.
The source manifest and gate summary are in this directory. All source
files in bin/, test/ and data/ still match that frozen hash; later edits
here are receipts and documentation only.

Read this directory's README, gate-summary.json, bind-departures.json,
x-reading-stamps.json, and frozen-source.json. Source changes are also
available in source-diff.patch. The adjacent grammar-contract receipt
lists all authorized fixture ports and retains the failed unstamped
cached-word fixture before its permitted order-one port. Its passes are
not promoted to this source. The live item-0 pending-premise certificate
and ForcedGrammar are unchanged. Existing assertions changed only in the
three lexical checks listed by the source audit under §§12–13.

Please check:

1. The grammar-only contract at all five §10 sites, both ReferenceClose
   recovery branches, PartRead.argument_reader and its callers: positions,
   reading order and occupations alone; identity fallback preserved; no
   new pronoun creation, grammar forcing or order-conditioned rule.
2. ReferenceReconstruction.cost and _sentence_path_cost: eligibility by
   the current closing item, same-close aliases, own-object/mint/gap/dummy
   zero, empty content zero, arrivals/thought episodes excluded; full
   description denominator when the candidate is narrower; preview-only
   records, no detach or live MTM read; independent weight 0.1; one active
   row vector with baseline one under the existing R enablement.
3. The new tests exercise the actual Error registry, SentenceCredit keep
   and score-function update and preserve rows/bank; weight/normalization
   and same-close aliases are covered. Check the scoped gate receipts
   against the source and the requested original replay inputs, and the
   fresh series' distinct first draws, bound/open counts and bind credit.
4. Any blocking correctness or contract issue. Distinguish it from the
   accepted, deferred root-position and re-identification work. Do not
   request a learning claim or a second sweep.

Give findings with source paths and line references, then a clear landing
recommendation. If a new design choice is needed, state the question for
Alec and your recommendation. The requested scope ends at line 1; step 2
and subsequent item-3.5 lines have not started.
