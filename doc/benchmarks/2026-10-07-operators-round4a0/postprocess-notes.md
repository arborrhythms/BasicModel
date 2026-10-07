# Saved-result accounting

`summarize.py`, `aggregate_saved.py` and `validate_saved.py` consume the saved
campaign only. They do not construct a model or add a forward/training pass.

The bounded sweep emits four successful reports for the single node ID
`test/test_use_flags.py::TestOrthogonalFlags::test_flags_match_expected`.
The frozen validator's raw report counter therefore totals 5,316 reports.
After that validator ran, `results-validation.json` was supplemented with
the case counts from `sweep-case-counts.json`: **5,313 distinct cases**, with
5,022 passes, 290 skips and one XPASS. The raw counter and duplicate reports
are preserved alongside the corrected case counter.

Validation also includes an explicit comparison of every requested 10/10
expectation, including the sum's final ¼ band. The older summarizer compares
the additive sum bar only; the receipt reports both criteria separately.
Neither supplementation changes a measured outcome or the frozen helpers.
