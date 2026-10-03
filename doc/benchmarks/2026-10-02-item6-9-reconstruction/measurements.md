# Closing measurements

Generated from first saved outcomes by `report_results.py`. No model is rerun.

| Campaign | Observed | Passes |
|---|---:|---:|
| class | 10/10 | 0/10 |
| reconstruction | 10/10 | 2/10 |
| sum | 10/10 | 10/10 |

Answers below follow `hello world`, `hello there`, `loving world`, `loving there`. Contrast is first + fourth − second − third.

## class

| Run | Four answers | MSE | Read-backs (same order) | Contrast | Seconds | Peak GiB |
|---:|---|---:|---|---:|---:|---:|
| 1 | 0.224018, 0.321309, 0.91219, 0.256402 | 0.146064 | hello world / hello there / loving world / loving there | -0.753079 | 312.226 | 1.42926 |
| 2 | 0.617631, 0.610758, 0.629669, 0.620309 | 0.263726 | world hello / there hello / loving loving / loving loving | -0.0024873 | 309.656 | 1.45186 |
| 3 | 0.356746, 0.455325, 0.46907, 0.39047 | 0.214573 | hello hello / there there / loving world / loving there | -0.177179 | 310.664 | 1.37943 |
| 4 | -0.0192734, 1.10371, 0.612123, 0.830151 | 0.212682 | hello world / hello hello / loving world / loving loving | -0.904958 | 308.514 | 1.44124 |
| 5 | 0.564064, 0.500754, 0.388043, 0.322758 | 0.26152 | world hello / there hello / world world / loving loving | -0.00197452 | 306.967 | 1.44145 |
| 6 | 0.234236, 0.606662, 0.822809, 0.0251336 | 0.0604024 | hello world / hello hello / loving world / loving loving | -1.1701 | 307.966 | 1.37625 |
| 7 | 0.634763, 0.667057, 0.620902, 0.617918 | 0.259828 | world world / there there / world world / there there | -0.0352783 | 306.963 | 1.37982 |
| 8 | 0.368993, 0.546571, 0.459955, 0.553414 | 0.234917 | hello hello / there there / loving loving / loving there | -0.0841203 | 310.212 | 1.46759 |
| 9 | 0.699824, 0.742306, 0.354665, 0.147299 | 0.248578 | hello hello / there there / world loving / loving there | -0.249848 | 308.111 | 1.44132 |
| 10 | 0.290319, 0.908033, 0.890101, 0.535885 | 0.0979983 | hello hello / hello there / loving world / loving there | -0.97193 | 307.49 | 1.44486 |
## reconstruction

| Run | Four answers | MSE | Read-backs (same order) | Contrast | Seconds | Peak GiB |
|---:|---|---:|---|---:|---:|---:|
| 1 | 0.592921, 0.594395, 0.474559, 0.476033 | 0.254692 | world world / there there / world world / there there | -8.9407e-08 | 305.337 | 1.44408 |
| 2 | 0.184181, 0.656135, 0.475181, 0.478073 | 0.164039 | world hello / hello hello / loving loving / loving loving | -0.469061 | 309.677 | 1.46344 |
| 3 | 0.324704, 0.660944, 0.72014, -0.00973237 | 0.0747021 | world hello / there hello / loving world / loving there | -1.06611 | 307.412 | 1.44016 |
| 4 | 0.479945, 0.60734, 0.280944, 0.215161 | 0.236966 | hello hello / hello hello / loving loving / there there | -0.193178 | 309.017 | 1.46674 |
| 5 | 0.538042, 0.425418, 0.606165, 0.439289 | 0.241929 | hello world / there there / loving world / loving loving | -0.0542509 | 305.748 | 1.44246 |
| 6 | 0.280264, -0.181113, 0.52825, -0.0752463 | 0.425447 | world world / there hello / loving world / loving there | -0.142119 | 303.791 | 1.44524 |
| 7 | 0.290668, 0.380213, 0.421357, 0.482406 | 0.259042 | world world / there there / world world / there there | -0.0284958 | 304.892 | 1.44106 |
| 8 | 0.620023, 0.406119, 0.658156, 0.527967 | 0.283182 | hello hello / hello hello / loving world / loving loving | 0.0837142 | 308.131 | 1.46807 |
| 9 | 0.279871, 0.517301, 0.909871, 0.294859 | 0.101598 | hello world / there hello / world world / there loving | -0.852441 | 308.573 | 1.44322 |
| 10 | 0.261768, 0.520097, 0.305605, 0.146575 | 0.200625 | hello world / hello there / loving world / loving there | -0.417358 | 310.744 | 1.37892 |
## sum

| Run | Four answers | MSE | Read-backs (same order) | Contrast | Seconds | Peak GiB |
|---:|---|---:|---|---:|---:|---:|
| 1 | 0.541587, 0.545621, 0.55061, 0.554644 | 0.252339 | hello hello / there there / world world / there there | 0 | 173.001 | 0.959505 |
| 2 | 0.518424, 0.536161, 0.536526, 0.554264 | 0.251481 | hello hello / hello hello / loving loving / loving loving | 5.96046e-08 | 173.428 | 0.960512 |
| 3 | 0.568895, 0.610046, 0.552832, 0.593982 | 0.25712 | world world / there there / world world / there there | 0 | 173.494 | 0.953707 |
| 4 | 0.576396, 0.561755, 0.509085, 0.494444 | 0.252441 | world world / there there / world world / there there | 0 | 174.041 | 0.960116 |
| 5 | 0.394128, 0.396551, 0.432144, 0.434566 | 0.257699 | hello hello / there there / loving loving / there there | 2.98023e-08 | 174.08 | 0.952425 |
| 6 | 0.580549, 0.55693, 0.600383, 0.576764 | 0.256425 | world world / there there / world world / loving loving | 0 | 171.878 | 0.95981 |
| 7 | 0.3418, 0.353811, 0.425181, 0.437192 | 0.263985 | world world / hello hello / world world / there there | 0 | 174.6 | 0.95888 |
| 8 | 0.505485, 0.487517, 0.502572, 0.484604 | 0.250107 | hello hello / hello hello / loving loving / loving loving | 0 | 172.918 | 0.958056 |
| 9 | 0.456702, 0.451067, 0.434387, 0.428752 | 0.253413 | hello hello / hello hello / loving loving / loving loving | 0 | 173.278 | 0.961626 |
| 10 | 0.59621, 0.605928, 0.567948, 0.577666 | 0.257782 | hello hello / hello hello / loving loving / there there | 0 | 173.268 | 0.953112 |

## Named XOR table

| Selector | Counts |
|---|---|
| `test/test_grounded_xor.py` | {'passed': 6} |
| `test/test_concept_output.py` | {'passed': 10} |
| `test/test_mm_xor.py` | {'passed': 6, 'failed': 1} |
| `test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp` | {'passed': 1} |
| `test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct` | {'passed': 1} |
| `test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy` | {'failed': 1} |
| `test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct` | {'failed': 1} |
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
| `test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy` | failed |
| `test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct` | failed |
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
| 1 | 900 | 1.26025e-09 | 5.60854e-07 | True | 1007.41 | 3.08117 |
| 2 | 900 | 1.77304e-08 | 1.89035e-08 | True | 1010.82 | 3.10082 |
| 3 | 900 | 7.03115e-06 | 1.97872e-05 | True | 996.967 | 3.25797 |
| 4 | 900 | 1.14575e-13 | 4.70735e-14 | True | 1013.32 | 3.095 |
| 5 | 900 | 0.000358709 | 0.00024099 | True | 1005.2 | 3.20769 |
| 6 | 900 | 2.06807e-05 | 3.47005e-05 | True | 1002.15 | 3.20887 |
| 7 | 900 | 2.37237e-10 | 1.67326e-09 | True | 1013.42 | 3.06092 |
| 8 | 900 | 0.250059 | 0.25005 | True | 1014.5 | 3.11455 |
| 9 | 900 | 1.53957e-11 | 1.09527e-10 | True | 965.052 | 3.22 |
| 10 | 900 | 4.22292e-07 | 1.37574e-07 | True | 893.535 | 3.21024 |

Median ending MSE: 2.20011e-07.

## Attribution

Ten fresh, unpaired unseeded runs per arm. R = reconstruction; E = expectation; A = answer. Without A the reader is not trained and the answer is omitted from trial comparison; its class result is diagnostic. Fixture, epoch budget and bars are unchanged. The receipt-local writer patches are saved in each run directory.

| Arm | Completed observations | Class bar | Reconstruction bar | Both existing bars |
|---|---:|---:|---:|---:|
| R | 10/10 | 0/10 | 1/10 | 0/10 |
| RE | 10/10 | 0/10 | 3/10 | 0/10 |
| RA | 10/10 | 1/10 | 1/10 | 0/10 |
| REA | 10/10 | 2/10 | 2/10 | 0/10 |

### R

| Run | Four answers | MSE | Read-backs | Contrast | Class / reconstruction | Seconds | Peak GiB |
|---:|---|---:|---|---:|---|---:|---:|
| 1 | 0.500921, 0.501333, 0.496252, 0.493749 | 0.249285 | hello world / hello there / loving loving / loving loving | -0.00291461 | False / False | 283.426 | 1.28458 |
| 2 | 0.498388, 0.487004, 0.494784, 0.484435 | 0.250369 | hello world / there hello / loving loving / there loving | 0.00103447 | False / False | 281.865 | 1.36652 |
| 3 | 0.486136, 0.486209, 0.495944, 0.496032 | 0.250108 | world world / there there / world loving / there loving | 1.58548e-05 | False / False | 279.782 | 1.28954 |
| 4 | 0.498873, 0.498918, 0.499358, 0.498932 | 0.249883 | hello hello / hello hello / loving loving / loving loving | -0.000470012 | False / False | 279.491 | 1.27744 |
| 5 | 0.502782, 0.507672, 0.498722, 0.503687 | 0.250039 | hello world / hello there / loving loving / loving there | 7.53403e-05 | False / False | 281.137 | 1.33267 |
| 6 | 0.496008, 0.496394, 0.495851, 0.496539 | 0.25009 | hello hello / hello hello / loving loving / loving loving | 0.000302374 | False / False | 281.125 | 1.36893 |
| 7 | 0.492797, 0.492643, 0.50682, 0.507028 | 0.250141 | world hello / there hello / loving world / loving loving | 0.000361621 | False / False | 282.928 | 1.37319 |
| 8 | 0.504336, 0.499959, 0.5105, 0.506212 | 0.250064 | world hello / there hello / loving world / there loving | 8.74996e-05 | False / True | 282.385 | 1.28537 |
| 9 | 0.49217, 0.49423, 0.491933, 0.495214 | 0.250351 | hello hello / there there / loving loving / loving there | 0.00122026 | False / False | 282.386 | 1.30489 |
| 10 | 0.505022, 0.497746, 0.497243, 0.502583 | 0.253165 | hello world / there there / loving world / there loving | 0.0126162 | False / False | 281.195 | 1.31363 |

### RE

| Run | Four answers | MSE | Read-backs | Contrast | Class / reconstruction | Seconds | Peak GiB |
|---:|---|---:|---|---:|---|---:|---:|
| 1 | 0.496853, 0.496454, 0.493739, 0.49336 | 0.250032 | world world / there there / world world / there there | 2.06232e-05 | False / False | 283.811 | 1.29383 |
| 2 | 0.51834, 0.508308, 0.514736, 0.510285 | 0.251577 | hello hello / there there / loving loving / loving there | 0.0055809 | False / False | 280.174 | 1.3528 |
| 3 | 0.504547, 0.504668, 0.500463, 0.501251 | 0.250178 | hello world / hello there / loving world / loving there | 0.000667632 | False / True | 280.172 | 1.33657 |
| 4 | 0.494948, 0.493479, 0.494902, 0.493312 | 0.250005 | hello hello / there hello / loving loving / there loving | -0.000120521 | False / False | 281.803 | 1.28542 |
| 5 | 0.497074, 0.485309, 0.497899, 0.483552 | 0.249479 | world world / hello there / world world / loving there | -0.00258219 | False / False | 283.441 | 1.29283 |
| 6 | 0.517923, 0.517675, 0.510056, 0.510049 | 0.250269 | world hello / there hello / loving world / loving there | 0.00024116 | False / True | 283.41 | 1.28859 |
| 7 | 0.506068, 0.50598, 0.507304, 0.506393 | 0.249836 | world world / there there / world loving / there loving | -0.000822723 | False / False | 282.856 | 1.37332 |
| 8 | 0.502154, 0.511296, 0.508402, 0.506933 | 0.24741 | world world / hello there / world world / there there | -0.0106124 | False / False | 283.934 | 1.30518 |
| 9 | 0.486995, 0.502318, 0.475059, 0.496423 | 0.251713 | world hello / there there / loving loving / loving there | 0.0060409 | False / False | 282.378 | 1.31748 |
| 10 | 0.496574, 0.492808, 0.498954, 0.493898 | 0.249703 | hello world / there hello / loving world / there loving | -0.00128895 | False / True | 280.759 | 1.37877 |

### RA

| Run | Four answers | MSE | Read-backs | Contrast | Class / reconstruction | Seconds | Peak GiB |
|---:|---|---:|---|---:|---|---:|---:|
| 1 | 0.280277, 0.607347, 0.789289, 0.439805 | 0.11764 | world world / hello there / world loving / there there | -0.676554 | False / False | 285.069 | 1.35863 |
| 2 | 0.745156, 0.639388, 0.678, 0.32009 | 0.22286 | world hello / there there / loving loving / loving there | -0.252142 | False / False | 282.446 | 1.38822 |
| 3 | 0.491997, 0.598864, 0.720867, 0.546011 | 0.194754 | world hello / there hello / loving loving / loving loving | -0.281722 | False / False | 284.074 | 1.30913 |
| 4 | 0.405885, 0.361589, 0.471393, 0.427098 | 0.258537 | hello hello / there hello / loving loving / loving loving | 5.96046e-08 | False / False | 282.482 | 1.29383 |
| 5 | 0.234805, 0.861663, 0.816619, 0.139536 | 0.0318423 | hello hello / there hello / world loving / there loving | -1.30394 | True / False | 283.426 | 1.30306 |
| 6 | 0.475697, 0.492192, 0.679078, 0.676426 | 0.261175 | hello world / hello there / world world / there there | -0.0191476 | False / False | 283.981 | 1.28957 |
| 7 | 0.40829, 0.42445, 0.362612, 0.349413 | 0.256578 | world hello / hello hello / world loving / loving loving | -0.0293587 | False / False | 282.352 | 1.39084 |
| 8 | 0.317194, 0.380542, 0.18806, 0.371472 | 0.320395 | hello world / hello there / loving world / loving there | 0.120064 | False / True | 282.361 | 1.29569 |
| 9 | 0.535618, 0.591648, 0.553542, 0.567415 | 0.24373 | hello hello / hello hello / loving loving / there there | -0.0421577 | False / False | 284.492 | 1.38536 |
| 10 | 0.369522, 0.35, 0.957713, 0.562897 | 0.219422 | hello hello / hello there / world loving / loving loving | -0.375294 | False / False | 283.922 | 1.3837 |

### REA

| Run | Four answers | MSE | Read-backs | Contrast | Class / reconstruction | Seconds | Peak GiB |
|---:|---|---:|---|---:|---|---:|---:|
| 1 | 0.490049, 0.578431, 0.977854, 0.344541 | 0.134267 | world hello / hello there / world loving / loving there | -0.721695 | False / True | 283.467 | 1.3795 |
| 2 | 0.640011, 0.628907, 0.655462, 0.612363 | 0.260255 | world world / there there / world world / loving loving | -0.0319945 | False / False | 284.033 | 1.31095 |
| 3 | 0.0133493, 0.865002, 0.559205, 0.248151 | 0.0685704 | world hello / there there / loving world / loving there | -1.16271 | False / False | 282.971 | 1.39078 |
| 4 | 0.560599, 0.971331, 0.722815, 0.541389 | 0.171257 | hello world / hello hello / world loving / loving loving | -0.592158 | False / False | 284.613 | 1.38912 |
| 5 | 0.328862, 0.812782, 0.746338, 0.376324 | 0.0872912 | hello world / hello there / world loving / loving there | -0.853934 | False / True | 281.919 | 1.38925 |
| 6 | 0.599555, 0.698814, 0.618009, 0.525479 | 0.218056 | world hello / there hello / loving loving / loving loving | -0.191789 | False / False | 281.915 | 1.36965 |
| 7 | 0.114785, 0.952564, 0.803365, 0.340865 | 0.04257 | hello world / hello there / world world / there loving | -1.30028 | True / False | 283.908 | 1.39292 |
| 8 | 0.177869, 0.926237, 0.836264, 0.13856 | 0.0207717 | hello hello / hello there / loving loving / loving there | -1.44607 | True / False | 283.905 | 1.30149 |
| 9 | 0.508548, 0.498492, 0.718054, 0.328845 | 0.174441 | world hello / there hello / world world / loving there | -0.379154 | False / False | 281.735 | 1.39316 |
| 10 | 0.432027, 0.753718, 0.676336, 0.134251 | 0.0925212 | world hello / there hello / loving loving / there loving | -0.863774 | False / False | 248.887 | 1.30411 |