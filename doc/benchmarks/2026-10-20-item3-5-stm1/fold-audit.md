# STM1 pre-implementation fold audit

No training, artifact loading, seed selection or gate relaxation. Native codes are read from the current model. Wider repeated-coordinate examples diagnose the existing kernel; they are not a proposed STM1 code.

For every width, intersection(a, b) = intersection(b, a), and intersection(a, a) = a. Exact ordered leaf-occurrence recovery therefore needs an order/multiplicity encoding as well as a width. The finite-bank clean-up index cannot resolve identical forward outputs. This does not reject invertible STM1; it identifies the missing structural law.

## Question 1: is the existing fold injective at width 232?

| Measurement | Result |
|---|---|
| width | 232 |
| native_width | 232 |
| widening | repeated coordinates for a diagnostic, not an STM1 implementation |
| ordered_inputs_distinct | True |
| swap_outputs_equal | True |
| duplicate_inputs_distinct | True |
| duplicate_outputs_equal | True |
| swap_max_error | 0.0 |
| duplicate_max_error | 0.0 |

## Question 2: is the existing fold injective at width 464?

| Measurement | Result |
|---|---|
| width | 464 |
| native_width | 232 |
| widening | repeated coordinates for a diagnostic, not an STM1 implementation |
| ordered_inputs_distinct | True |
| swap_outputs_equal | True |
| duplicate_inputs_distinct | True |
| duplicate_outputs_equal | True |
| swap_max_error | 0.0 |
| duplicate_max_error | 0.0 |

## Question 3: is the existing fold injective at width 1856?

| Measurement | Result |
|---|---|
| width | 1856 |
| native_width | 232 |
| widening | repeated coordinates for a diagnostic, not an STM1 implementation |
| ordered_inputs_distinct | True |
| swap_outputs_equal | True |
| duplicate_inputs_distinct | True |
| duplicate_outputs_equal | True |
| swap_max_error | 0.0 |
| duplicate_max_error | 0.0 |

## Question 4: is the existing fold injective at width 7424?

| Measurement | Result |
|---|---|
| width | 7424 |
| native_width | 232 |
| widening | repeated coordinates for a diagnostic, not an STM1 implementation |
| ordered_inputs_distinct | True |
| swap_outputs_equal | True |
| duplicate_inputs_distinct | True |
| duplicate_outputs_equal | True |
| swap_max_error | 0.0 |
| duplicate_max_error | 0.0 |
