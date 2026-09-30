| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.45 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.86 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.86 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.86 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.86 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.86 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.86 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.86 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.86 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.86 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.86 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.96 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.96 GiB | ending 0.19906657934188843, calls 18, predictions [0.46813708543777466, 0.5787351727485657, 0.5778230428695679, 0.47054916620254517] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.96 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.96 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.96 GiB | ending 0.2500185966491699, calls 1, predictions [0.4956045150756836, 0.4956114590167999, 0.49548542499542236, 0.4954874515533447] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.96 GiB | ending 0.1993352472782135, calls 59, predictions [0.4377530813217163, 0.5597696304321289, 0.5605068206787109, 0.46771377325057983] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.96 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | failed; peak 1.29 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | failed; peak 1.29 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB |  |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB |  |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.00 GiB | exact 1.0; where 1.0; output 0.17501772940158844; reconstruction 0.00013363624748308212 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.84 GiB | exact 1.0; where 1.0; output 0.17500759661197662; reconstruction 0.0001300703443121165 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 3.99 GiB | exact 1.0; where 1.0; output 0.17501485347747803; reconstruction 0.00013815879356116056 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 3.74 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.09 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.04 GiB | exact 1.0; where 1.0; output 0.17700491845607758; reconstruction 0.00016337446868419647 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.12775886058807373; reconstruction 0.00021168586681596935 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17400936782360077; reconstruction 3.091711550951004e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1672435998916626; reconstruction 4.763592733070254e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.16597305238246918; reconstruction 3.912103420589119e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.16761979460716248; reconstruction 6.016954284859821e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.1750417798757553; reconstruction 1.7150705389212817e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.1798226684331894; reconstruction 2.9868941055610776e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.1774667203426361; reconstruction 1.2436980796337593e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17500834167003632; reconstruction 1.4646341696789023e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17663712799549103; reconstruction 4.3374493543524295e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17034253478050232; reconstruction 6.245195254450664e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17654119431972504; reconstruction 2.2268417524173856e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.16503575444221497; reconstruction 7.48617821955122e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17492729425430298; reconstruction 2.3964517822605558e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17527616024017334; reconstruction 7.892993016866967e-05 |
