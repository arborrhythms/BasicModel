# Item 12: expectation review corrections

Baseline: `d5b6cad` (including its documentation changes), after the accepted
expectation mechanism in `7d7dc4f`. This landing changes retention normalization,
pair lookup, purity coverage and diagnostics. It does not claim a learning gain;
the previous null utility result remains open under countdown item 9.

## Corrections

- Retention uses the soft union of observed occupancy and predicted presence:
  `w = observed_mask + (1 - observed_mask) * presence`. The scalar before
  `s/(1+s)` is `sum(w * mean((o-e)^2, coordinates)) / sum(w)`, or zero with
  no roles in play. Equal per-role error gives equal surprise for idea and
  relation rows. Missing expected roles count without a hard presence cutoff.
  The predictor's all-role MSE, presence BCE and raw residual are unchanged.
- The native purity snapshot includes nonuniform priority and live reading
  scope, before and after sealing. The reading readout is nonzero, so the
  cursor bootstrap cannot hide a content leak. Staged/unstaged estimates,
  gains zero/one and disabled expectation yield bit-identical snapshots.
- Pair lookup uses the existing occurrence index in both directions. Tests
  reject access to unrelated rows, including after restore and compaction.
- The bounded harness recognizes a typed stale-PCH `CppCompileError`, including
  Torch backend wrappers. It retries the affected case once in a fresh worker
  with PCH reuse disabled. Compilation remains active and the shared cache is
  untouched. Receipt events, first-attempt logs and raw JSON preserve the
  failure. Assertions, other compiler errors, mixed teardown failures and
  failures on retry remain failures. Injection tests exercise actual subprocess
  receipts; the classifier also checks real Torch exception classes.
- Operator diagnostics log output/reconstruction and expectation/reconstruction
  gradient norm ratios beside cosine. Zero reconstruction gives a null ratio;
  zero other gradient with nonzero reconstruction gives zero. A 2,400-fold
  aligned gradient is explicitly visible. This report does not modify gradients.

## Development evidence

The [initial probes](initial-red.txt) expose the missing normalization and ratio
and the absent cache classifier. That first probe also caught an invalid test
fixture's origin keyword; after correcting the fixture's origin setup,
the [indexed-read probe](index-red.txt) fails on the actual unrelated-row scans
for live, restored and compacted stores. The [first focused run](initial-focused-green.txt)
passes all 47 cases. The final affected and full receipts below supersede that
run after strengthening the reading readout and compiler-wrapper checks.

The final affected run, `20260921-063236-32ba94`, completes **131/131** selected
cases: **128 passed, 3 skipped**, exit 0, in 218.58 seconds. Its
[receipt](affected-result.json.gz), [source manifest](affected-source-manifest.json)
and [run log](affected-run.txt) cover the expectation, retention, gradient,
joint-objective, diagnostic-contract and bounded-runner files.

The [native operator report](operator-gradients.json) records
`operator.CS.surface`: reconstruction norm `0.00074938984`, output norm
`1.82951338`, output/reconstruction ratio **2441.3373**, cosine **0.08721979**.
Expectation has no gradient on that operator in this supplied-lesson probe.
These are magnitudes from one diagnostic batch, not evidence of persistent
opposition or useful learning.

## Full receipt

`20260921-063627-f5a587` completes **4,683/4,683** selected cases:
**4,352 passed, 330 skipped, 1 expected failure**, exit **0**, in **1,059.92 s**.
No failure was waived and no compile-cache retry was needed in this full run.
The formerly failing `test_normal_batch_trains_supplied_grammar_lessons` passes.
The skip and expected-failure gates are unchanged.

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

The existing limits remain: ten workers, 8 GiB per worker and 28 GiB aggregate;
peak aggregate footprint was **14.34 GiB**. Eight-case/one-file batches avoid
the accumulation issue still assigned to countdown item 2.

- [Full result](full-result.json.gz), [run log](full-run.txt),
  [source manifest](full-source-manifest.json), [receipt metadata](receipt-info.json).
- Both affected and full runs match **629 source files**, fingerprint
  `ec2a3c999c5e3c5c26de2ca2034fbf2195cc55c82f5e47628fd1eb652f06f034`
  (SHA-256 of the sorted compact JSON `validated_source` map).
- Counts are unique selected cases. The unchanged orthogonal-flags test emits
  five passing call reports for one node; all reports remain in the receipt,
  and the metadata records this separately from case counts.

No source was edited during either bounded run. Documentation and the final
receipt were reconciled afterward; the tested source fingerprint is unchanged.

Final documentation-link run `20260921-065608-165748` passes **68/68** cases
with the same validated source ([receipt](doc-links-result.json.gz),
[run log](doc-links-run.txt)).
