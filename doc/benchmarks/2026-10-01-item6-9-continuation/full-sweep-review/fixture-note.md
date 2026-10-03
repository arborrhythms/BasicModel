# Sentence-state fixture mismatch during the frozen full sweep

`test_compiled_expectation_boundary.py::test_explicit_boundary_retains_factored_roles_and_skips_masked_rows`
failed because `test/reading_fixtures.py::commit_reading` builds the pre-step-7
private CSLang tuple. `_discard_sentence_record` now clears indices 26 and 27
(the per-slot last unary and per-round alternative mask), so its access to the
older tuple raises IndexError. The fixture must be ported with those tensor
fields, preserving every behavioral assertion. The public 21-value compiled
result is unchanged; the real compiled forward/backward check passed before
this sweep.

This is an unresolved fixture port in the tested candidate. No test or runtime
file was edited during the single frozen full sweep. Every affected case will
be listed in the final failure table, rather than counted as passing.

The direct tuple builder in `test_item7_end_state_storage.py::test_closing_discards_operations_without_changing_other_live_rows` has the same missing-field problem; it does not call the shared reading fixture.
