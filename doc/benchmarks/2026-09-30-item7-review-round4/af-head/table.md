| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.46 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.46 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.92 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.92 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.92 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.92 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.92 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.92 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.92 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.92 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.92 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.92 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 1.41 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 1.41 GiB | ending 0.1850692331790924, calls 18, predictions [0.4067487418651581, 0.5534712672233582, 0.5534712672233582, 0.41959095001220703] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 1.41 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 1.41 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 1.41 GiB | ending 0.2500050663948059, calls 1, predictions [0.505202054977417, 0.5052273273468018, 0.5053248405456543, 0.5052599906921387] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 1.41 GiB | ending 0.1991976648569107, calls 116, predictions [0.45123884081840515, 0.55165696144104, 0.5522019267082214, 0.4377666115760803] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 1.41 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | failed; peak 1.29 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | failed; peak 1.30 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB |  |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB |  |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 3.99 GiB | exact 1.0; where 1.0; output 0.17501772940158844; reconstruction 0.00013363624748308212 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.82 GiB | exact 1.0; where 1.0; output 0.17500759661197662; reconstruction 0.0001300703443121165 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 3.99 GiB | exact 1.0; where 1.0; output 0.17501132190227509; reconstruction 0.00013247497554402798 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 3.73 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.08 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.05 GiB | exact 1.0; where 1.0; output 0.17673829197883606; reconstruction 0.00013022632629144937 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.18163032829761505; reconstruction 0.00013781417510472238 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.17438605427742004; reconstruction 3.4777240216499195e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17575038969516754; reconstruction 3.97556068492122e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.17136482894420624; reconstruction 4.349618393462151e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.17597317695617676; reconstruction 9.796010272111744e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17717023193836212; reconstruction 1.8774142517941073e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17643533647060394; reconstruction 7.543805986642838e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17423047125339508; reconstruction 4.48933060397394e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | failed; peak 4.18 GiB | exact 0.75; where 1.0; output 0.1769406646490097; reconstruction 2.0799338017241098e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17522932589054108; reconstruction 2.225632306362968e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.1794147938489914; reconstruction 2.2796495613874868e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.1753639429807663; reconstruction 5.225411314313533e-06 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17314530909061432; reconstruction 3.524194471538067e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.18105758726596832; reconstruction 0.00011347161489538848 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.17477385699748993; reconstruction 7.3362230068596546e-06 |
