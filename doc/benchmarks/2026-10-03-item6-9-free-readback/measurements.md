# Closing measurements

Generated from first saved outcomes by `report_results.py`. No model is rerun.

| Campaign | Observed | Passes |
|---|---:|---:|
| class | 10/10 | 9/10 |
| reconstruction | 10/10 | 5/10 |
| sum | 10/10 | 10/10 |

Answers below follow `hello world`, `hello there`, `loving world`, `loving there`. Contrast is first + fourth − second − third.

## class

| Run | Four answers | MSE / band | Read-backs (same order) | Contrast | Seconds | Peak GiB |
|---:|---|---:|---|---:|---:|---:|
| 1 | -7.7486e-07, 0.999999, 0.999999, -1.54972e-06 | 1.0667e-12 / at 0 | world hello / hello there / loving loving / loving loving | -2 | 217.556 | 1.13341 |
| 2 | -0.0154337, 0.931022, 1.01145, -0.0617124 | 0.00223391 / at 0 | hello world / hello there / world loving / there loving | -2.01962 | 209.594 | 1.15254 |
| 3 | 4.47035e-07, 1, 1, 1.49012e-07 | 2.40252e-13 / at 0 | world world / hello hello / world world / loving loving | -2 | 210.6 | 1.07717 |
| 4 | -0.00209284, 1.00476, 0.999458, 0.00574669 | 1.5088e-05 / at 0 | world hello / hello there / world world / there there | -2.00056 | 208.992 | 1.07775 |
| 5 | 0.0716745, 1.00257, 0.93162, 0.00998774 | 0.00247986 / at 0 | world hello / there hello / loving world / there loving | -1.85253 | 207.436 | 1.15067 |
| 6 | 0.0380442, 0.553314, 0.866972, -0.112503 | 0.0578323 / between | hello world / there hello / loving world / there loving | -1.49475 | 206.288 | 1.15489 |
| 7 | -0.000573933, 1.00092, 1.00209, 0.00115767 | 1.72711e-06 / at 0 | hello world / there hello / world world / loving loving | -2.00243 | 208.482 | 1.08098 |
| 8 | 2.27988e-05, 1.00002, 0.999994, -2.64049e-05 | 4.10165e-10 / at 0 | hello world / there hello / loving loving / there there | -2.00002 | 209.018 | 1.07257 |
| 9 | -0.0145888, 0.970404, 0.99473, 0.0126696 | 0.000319266 / at 0 | world hello / there hello / world world / there there | -1.96705 | 207.932 | 1.07561 |
| 10 | 0.0820876, 0.99686, 1.02428, 0.00553364 | 0.00184206 / at 0 | world hello / hello there / loving world / loving there | -1.93352 | 206.887 | 1.15555 |
## reconstruction

| Run | Four answers | MSE / band | Read-backs (same order) | Contrast | Seconds | Peak GiB |
|---:|---|---:|---|---:|---:|---:|
| 1 | -0.00336093, 0.990695, 1.00392, -0.00666124 | 3.9394e-05 / at 0 | hello world / there hello / world loving / loving there | -2.00463 | 205.807 | 1.08261 |
| 2 | -0.0121064, 0.9895, 0.998713, 0.00439048 | 6.94354e-05 / at 0 | world world / there there / world world / there there | -1.99593 | 209.053 | 1.08547 |
| 3 | 1.46031e-06, 1.00001, 0.99999, -7.09295e-06 | 4.62965e-11 / at 0 | world hello / hello there / world loving / loving there | -2 | 207.337 | 1.15306 |
| 4 | 0.00633383, 1.00566, 1.01042, -0.0128455 | 8.64116e-05 / at 0 | world hello / there hello / loving loving / there there | -2.02259 | 208.338 | 1.0932 |
| 5 | 0.0653913, 0.62603, 0.91857, -0.387345 | 0.0751991 / between | hello world / there hello / world loving / there loving | -1.86655 | 207.801 | 1.08109 |
| 6 | 0.0079208, 1.01112, 0.994618, 0.0067271 | 6.51729e-05 / at 0 | hello world / there hello / world loving / loving there | -1.99109 | 210.068 | 1.15499 |
| 7 | 2.98023e-07, 1, 1, 1.49012e-07 | 2.58682e-13 / at 0 | world world / hello hello / world loving / loving there | -2 | 208.516 | 1.0856 |
| 8 | -0.000970602, 0.976005, 1.02149, -8.91685e-05 | 0.000259643 / at 0 | hello world / there hello / world loving / there loving | -1.99856 | 208.995 | 1.15505 |
| 9 | 0.366528, 0.823972, 0.799777, -0.223693 | 0.063864 / between | world world / there there / world loving / there loving | -1.48091 | 209.031 | 1.09944 |
| 10 | 0.000322044, 1.00217, 0.998577, 0.000764489 | 1.85792e-06 / at 0 | world world / hello hello / loving world / loving there | -1.99966 | 210.655 | 1.07915 |
## sum

| Run | Four answers | MSE / band | Read-backs (same order) | Contrast | Seconds | Peak GiB |
|---:|---|---:|---|---:|---:|---:|
| 1 | 0.499217, 0.498923, 0.499108, 0.498814 | 0.250001 / at 1/4 | hello world / hello there / loving world / loving there | 0 | 143.771 | 0.873339 |
| 2 | 0.499973, 0.49998, 0.499971, 0.499978 | 0.25 / at 1/4 | hello world / hello there / world loving / there loving | 0 | 144.779 | 0.882647 |
| 3 | 0.495056, 0.494321, 0.506852, 0.506117 | 0.250035 / at 1/4 | world hello / hello there / world loving / loving there | -2.98023e-08 | 144.766 | 0.871264 |
| 4 | 0.517572, 0.521669, 0.508374, 0.512471 | 0.250251 / at 1/4 | world hello / there hello / world loving / loving there | -5.96046e-08 | 144.231 | 0.873156 |
| 5 | 0.499993, 0.499974, 0.499981, 0.499962 | 0.25 / at 1/4 | world hello / hello there / world loving / there loving | -2.98023e-08 | 144.782 | 0.878527 |
| 6 | 0.498677, 0.498838, 0.498824, 0.498986 | 0.250001 / at 1/4 | hello world / hello there / world loving / loving there | 0 | 143.742 | 0.886873 |
| 7 | 0.499999, 0.499999, 0.5, 0.499999 | 0.25 / at 1/4 | hello world / hello there / world loving / loving there | 0 | 144.823 | 0.871584 |
| 8 | 0.499585, 0.499682, 0.499453, 0.49955 | 0.25 / at 1/4 | hello world / hello there / world loving / loving there | 2.98023e-08 | 144.317 | 0.881136 |
| 9 | 0.499855, 0.500842, 0.500392, 0.501379 | 0.250001 / at 1/4 | hello world / there hello / loving world / there loving | -2.98023e-08 | 143.705 | 0.867067 |
| 10 | 0.499999, 0.499999, 0.499999, 0.499999 | 0.25 / at 1/4 | hello world / there hello / world loving / loving there | 0 | 143.704 | 0.876131 |

## Named XOR table

| Selector | Counts |
|---|---|
| `test/test_grounded_xor.py` | {'passed': 6} |
| `test/test_concept_output.py` | {'passed': 10} |
| `test/test_mm_xor.py` | {'passed': 6, 'failed': 1} |
| `test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp` | {'passed': 1} |
| `test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct` | {'passed': 1} |
| `test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy` | {'passed': 1} |
| `test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct` | {'passed': 1} |
| `test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor]` | {'passed': 1} |
| `test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise]` | {'passed': 1} |
| `test/test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live` | {'passed': 1} |
| `test/test_reconstruction_roundtrip.py::test_xor_recon_grads_flow` | {'passed': 1} |
| `test/test_reconstruction_roundtrip.py::test_xor_percepts_tile_words` | {'passed': 1} |
| `test/test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget` | {'passed': 1} |
| `test/test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip` | {'passed': 1} |

### Every named case

| Case | Outcome |
|---|---|
| `test/test_grounded_xor.py::test_native_unseeded_xor[0-4]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[0-8]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[1-4]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[1-8]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[2-4]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[2-8]` | passed |
| `test/test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown` | passed |
| `test/test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field` | passed |
| `test/test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence` | passed |
| `test/test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower` | passed |
| `test/test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not` | passed |
| `test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0]` | passed |
| `test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1]` | passed |
| `test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2]` | passed |
| `test/test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent` | passed |
| `test/test_concept_output.py::test_property_reverse_attributes_only_written_members` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_convergence` | failed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_model_is_mental` | passed |
| `test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp` | passed |
| `test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct` | passed |
| `test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy` | passed |
| `test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct` | passed |
| `test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor]` | passed |
| `test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise]` | passed |
| `test/test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live` | passed |
| `test/test_reconstruction_roundtrip.py::test_xor_recon_grads_flow` | passed |
| `test/test_reconstruction_roundtrip.py::test_xor_percepts_tile_words` | passed |
| `test/test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget` | passed |
| `test/test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip` | passed |

## MM_grammar

| Run | Completed epochs | Ending training MSE | Final evaluation MSE | Complete | Seconds | Peak GiB |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 900 | 7.11064e-07 | 9.53458e-07 / at 0 | True | 522.916 | 2.19425 |
| 2 | 900 | 4.28111e-11 | 5.63194e-12 / at 0 | True | 517.288 | 2.02877 |
| 3 | 900 | 1.06581e-14 | 4.44089e-15 / at 0 | True | 522.717 | 2.28717 |
| 4 | 900 | 2.24161e-08 | 2.26974e-08 / at 0 | True | 517.542 | 2.02974 |
| 5 | 900 | 1.15497e-05 | 1.47985e-05 / at 0 | True | 517.037 | 2.02568 |
| 6 | 900 | 9.08837e-09 | 6.30791e-09 / at 0 | True | 518.177 | 2.21918 |
| 7 | 900 | 1.09144e-07 | 2.98319e-07 / at 0 | True | 524.207 | 2.00438 |
| 8 | 900 | 0.25 | 0.25 / at 1/4 | True | 515.532 | 2.23828 |
| 9 | 900 | 1.2217e-06 | 2.21816e-08 / at 0 | True | 514.356 | 2.03609 |
| 10 | 900 | 0.000204326 | 0.000228986 / at 0 | True | 456.624 | 2.24111 |

Median ending MSE: 4.10104e-07.
