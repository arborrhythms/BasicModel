# Part diffusion in step 1b

The current production channel reads the native Part pairs and the
word/sentence memberships derived from those pairs. The default
`primingSpread` is `.05`. Both reads spend one degree-normalized budget;
only the priming surface is changed. This channel does not read code
placement, form coordinates or a geometric similarity matrix.

The newer [Orders §9](../../Orders.md) amendment places the order strata,
order-0 geometric channel and code placement in **Item 3**. The earlier
geometric draft is preserved as
[implementation](draft_geometric_diffusion.py),
[tests](draft_geometric_diffusion_tests.py) and
[probe](draft_geometric_diffusion_probe.py).
Measurements before native-61 describe that draft and are not promoted
to the present Part-only implementation.

## 1. Is the diffusion budget conservative?

| Question | Result |
|---|---|
| Multiple channels | Part edges and derived memberships share each source's normalized outflow |
| Multiple inhibited neighbors | The stream spends only the feasible fraction of the common budget, so the standing floor cannot create energy at a hub |
| Inactive streams | Unchanged |
| Code bank / gradients / random state | Not read or changed by the new graph channel |
| Native address lifetime | The graph is read from the current native store after loading and forgetting |
| Regression result | All three `test_priming_diffusion.py` cases pass in native-70 |

## 2. Does native sentence membership activate other words?

| Measurement | Native-70 |
|---|---:|
| Fixture | `test_actual_occurrences_activate_words_outside_the_current_sentence` |
| Diffusion events | 2 |
| Maximum directed Part edges | 8 |
| Maximum membership edges | 8 |
| Maximum membership rows | 4 |
| Budget-limited streams | 0 |
| Maximum total-energy error | 0 |

The [native trace](priming-70/test_actual_occurrences_activate_words_outside_the_current_sentence.json)
records actual owner projections. It does not infer activation from a
synthetic matrix.

## 3. What happens during the native §14 initialization read?

| Measurement | Native-70 |
|---|---:|
| Diffusion events | 15 |
| Membership events | 14 |
| Maximum directed Part edges | 386 |
| Maximum membership edges | 151 |
| Maximum membership rows | 116 |
| Budget-limited streams | 0 |
| Maximum total-energy error, summed in double | 0.0000035762786865234375 |

See the [read-only probe output](priming-70/test_ordinary_initial_binding_distribution_includes_every_retained_candidate.json).
The small error is float32 transfer roundoff, measured with double totals
so the neutral inventory does not hide it. These measurements establish
mechanism behavior only. They do not establish learned reference
accuracy, collapse rates or a graded-versus-shuffled result.
