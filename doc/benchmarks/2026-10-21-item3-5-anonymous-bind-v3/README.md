# Step 1b §10.4: closing bind and the decomposition open-name seam

Frozen source: `3d453e6bb6927e5da1492c25106a4ef107b87bca`.
All requested gates have completed on this hash. Publication is blocked
by the failures below; [Claude's completed review](claude-review.md) says
not to land. No failed draw is
replaced, no earlier-source pass is promoted, and no second full sweep
or learning claim is made.

Production is unchanged from `823f157cd`. This freeze also connects
`ForcedDecompositionGrammar`'s existing `open_names=('what',)` rule to
`ReferenceClose.select`, so an arrival bind cannot prefill its question.
The original snapshots, seeds, thought requests, reference preferences,
assertions, budgets, objective and live keep remain unchanged. Length 1
passes its original zero-arm snapshot in the development replay.

| Gate on this source | Result | Evidence |
|---|---|---|
| Twelve distinct first draws, six per order | 10 passed, reverse 2 and 3 failed D2; both exactly replay | [Full ledger and captures](fresh-final.md), [draw 2](reverse-2-failure.md), [draw 3](reverse-3-failure.md) |
| Eight-file set | 110 passed, 3 failed, 1 existing copular prerequisite skip | [Failures and diagnoses](native-failures.md), [bounded report](native-final/report.html) |
| MTM, anonymous arrival, Part closing, occurrence views | 48 passed (14 + 16 + 14 + 4), including eager and compiled isolation | [Bounded report](regressions-final/report.html) |
| Both unchanged merge certificates | Pass; one working group, with 2 / 4 separate pool entries | `native-final-reference-evidence` |
| Both original decomposition arms, both lengths | 4 passed; all 8 companion bindings and proof chains correct | [Every question](decomposition-final-questions.md) |
| Eight-seat MM, rng-119 | Pass; zero seat requests and lower edges | [Run](mm-saved-final/report.html), [capacity trace summary](mm-saved-final-state/summary.json) |
| Eight-seat MM, same captured fresh draw | Pass; zero seat requests and lower edges | [Run](mm-fresh-final/report.html), [capacity trace summary](mm-fresh-final-state/summary.json) |
| Source audit | All 23 bounded runs match all 825 frozen source files; frozen helpers match | [Audit](source-audit.json), [unchanged contract files](unchanged-contract-files.json) |

The [A decomposition](reference-cost-breakdown.md) confirms the earlier
deciding step in the reverse failures: R ties; the named path pays
.407944173 in closing reference credit, while the kept field/mint reading
pays zero and creates no pending x site. Its retained order-0 three
complement is then absent from the next frame menu, preventing recovery.
The native set adds an invalid nested occurrence
witness from `bind_site` and the unchanged bootstrap zero-R failure.
`witness-repro-04` isolates the witness error. All are mechanism failures
or unresolved contract questions, never passing ports. The bootstrap
case is captured from initialization on this source; the earlier
[accepted-source comparison](../2026-10-21-item3-5-anonymous-bind-v2/bootstrap-failure.md)
is diagnostic evidence only and does not waive it.

The [previous source's full ledger](../2026-10-21-item3-5-anonymous-bind-v2/README.md)
retains its twelve fresh passes, 48 regression passes, eight-file failure
and all four decomposition failures. Read-only post-freeze diagnostics
are separately [hashed](diagnostic-helpers.json). Production, existing
tests, targets, forcing scope, objective and keep law remain unchanged
after this freeze. A one-line witness repair passes five cases in an
isolated development copy (`diagnostics/runs/witness-repair-05`); that is
not a frozen-source pass. The [review request](claude-review-request.md)
and [response](claude-review.md) are retained in full. The later bootstrap
diagnostic [extends the comparison to this exact source](bootstrap-reviewed-source-comparison.json).
The forcing/objective decision is pending with Alec; neither is changed.
