# Standalone probes, diagnostics, and micro-benchmarks (not pytest-collected).
# Moved out of test/ in the 2026-07-17 cleanup (Tier-2 item 4) so test/ holds
# only the suite and its shared helpers (conftest, fixtures/, _stm_test_fixtures,
# space_equiv). Run directly, e.g. `PYTHONPATH=bin python test/tools/bench_training.py`.
