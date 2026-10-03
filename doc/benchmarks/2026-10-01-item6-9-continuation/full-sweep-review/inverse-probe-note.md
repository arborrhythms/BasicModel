# Missed inverse-probe port

`test_tied_operator_reconstruction.py::test_missing_absorbed_operand_uses_bounded_compose_candidates`
uses the asymmetric `PartLayer`. It expects the missing left operand to be the
mean of the two valid candidates, and the right operand to remain the parent.
Step 6 replaces the blended forward candidate with one minimum-residual hard
candidate, retaining the soft gradient. The new focused symmetric-pair and
gradient probe passed, but this existing absorbed-operand probe was not ported
before the sweep.

A port must retain the available-inverse assertion, the unchanged right-parent
assertion and the exclusion of the invalid candidate. It should assert that
the left is a valid minimum-residual candidate, without claiming that this
information-losing operation recovers a unique marker identity. The old test
contains no gradient assertion; adding one may document the retained soft
surrogate. Complete old/new bodies would be required when that port is made.
The frozen candidate was not edited after this failure.
