| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.45 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.83 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.83 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.83 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.83 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.83 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.83 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.83 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.83 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.83 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.83 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.66 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.66 GiB | ending 0.19538599252700806, calls 18, predictions [0.449005663394928, 0.5536542534828186, 0.5780819654464722, 0.45022052526474] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.66 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.66 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.66 GiB | ending 0.2500959634780884, calls 1, predictions [0.5050426721572876, 0.5048571228981018, 0.5047858357429504, 0.5048883557319641] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.66 GiB | ending 0.19919662177562714, calls 22, predictions [0.4588578939437866, 0.5440696477890015, 0.5723762512207031, 0.4421553909778595] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.66 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | failed; peak 1.29 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | failed; peak 1.29 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB |  |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB |  |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 3.99 GiB | exact 1.0; where 1.0; output 0.17501772940158844; reconstruction 0.00013363624748308212 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.84 GiB | exact 1.0; where 1.0; output 0.17500759661197662; reconstruction 0.0001300703443121165 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.00 GiB | exact 1.0; where 1.0; output 0.17501266300678253; reconstruction 0.00014600878057535738 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 3.73 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.07 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 3.99 GiB | exact 1.0; where 1.0; output 0.17625118792057037; reconstruction 0.00015049854118842632 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.18027383089065552; reconstruction 9.526441863272339e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.16367945075035095; reconstruction 0.0001728638744680211 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17007572948932648; reconstruction 0.00013821799075230956 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17035190761089325; reconstruction 0.00014729677059222013 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.1816386729478836; reconstruction 6.924240005901083e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | failed; peak 4.17 GiB | exact 0.75; where 1.0; output 0.17571091651916504; reconstruction 1.7893569747684523e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.2487134039402008; reconstruction 0.00035027373814955354 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.1330050379037857; reconstruction 0.00017560833657626063 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.1751258820295334; reconstruction 4.959799844073132e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17524829506874084; reconstruction 4.379899110062979e-06 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.18599402904510498; reconstruction 7.694704254390672e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.17643888294696808; reconstruction 1.6994730685837567e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17521420121192932; reconstruction 3.307299630250782e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17737466096878052; reconstruction 2.3451260858564638e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.18021327257156372; reconstruction 2.7185325961909257e-05 |
