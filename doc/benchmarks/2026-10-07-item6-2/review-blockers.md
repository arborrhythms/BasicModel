# Review blockers from the frozen item 6.2 measurements

This candidate is **not ready to land**. The default sweep passed, but the
required explicit thinking checks did not. Source and tests remain at the
measured hash; no failed training or gate was retried.

## Unforced MM_query_reasoning

The configured run completed two of 300 training epochs before an
`IndexError` in `MemoryIndex._meaning_leaf_terms`, reached through
`ThoughtStream.write`. It opened 12 root thought episodes. Seven credit
observations were recorded, none carrying a policy gradient before
the failure. This is not a learned-thinking success.

The [saved outcome](mm-query-configured/outcome.json) contains the full trace.
The failing meaning itself was not captured. A separate
[construction-only diagnostic](constituent-diagnostic.json) reproduces the
same exception without a model training, a seed, or an episode/gate rerun:
`ThoughtReferences.fill` copies `('constituent', 0)` from the supplied
meaning into a goal that has no constituents. The supplied meaning has one
child; the filled meaning has none. `ThoughtStream.write` then indexes the
missing child. The current local-reference preflight checks LTM occurrences,
not the bounds of a meaning's local constituent table.

The review should resolve ownership when a binding transfers a local child:
carry and rebase its constituent graph, then validate local references before
indexing or writing. The diagnostic establishes this defect; it does not prove
that this was the only route to the original configured run's failure.

## Explicit thinking mechanism sweep

[All 47 selected cases completed](thinking-gate/result.json): 41 passed and
six failed. The groups are:

| Group | Passed | Failed |
| --- | ---: | ---: |
| New 6.2 certificates, including chaining and answer/expectation credit | 25 | 0 |
| Normal per-row answer-credit integration | 1 | 0 |
| MM_query_reasoning configuration and supplied-reading optimizer smoke | 1 | 1 |
| Historical 11c explicit mechanism selection | 14 | 5 |

The MM optimizer smoke failed with incompatible routing dimensions:
`mat1 and mat2 shapes cannot be multiplied (6x58 and 11x104)` in
`BracketExpectation._apply_routing_bias`.

The five 11c failures concern an out-of-range percept ID, the expected view
width, universe routing, `_stage0_indices`, and `_primed_reading_step`.
Those three test files are byte-identical to the accepted 6.5 landing.
[Failure messages and file hashes](thinking-failures.json) are retained.
This provenance does not establish whether each failure predates 6.2; no
baseline training rerun was performed. The old local `torch.manual_seed`
calls were suppressed and recorded for the requested unseeded measurement.

The thinking launcher also has a post-report formatting error: it treats the
path returned by `bounded_tests.main` as a result dictionary. This occurs
**after** the complete `thinking-gate/result.json` and worker logs are saved;
it neither causes nor masks the six test failures. The executed launcher is
preserved in the measurement-helper archive. No gate was rerun to repair its
reporting.

## Review boundary

The thirty standing trainings have completed under their original assertions
and configuration files. Their complete outcomes, including any misses, are
in the main receipt. The source remains frozen for that measurement. These
findings are for Claude's review before any commit; neither the 6.2 landing nor
any million-sentence learning claim is accepted here.
