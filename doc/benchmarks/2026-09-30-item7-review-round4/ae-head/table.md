| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.45 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.45 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.91 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.91 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.91 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.91 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.91 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.91 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.91 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.91 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.91 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.91 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.77 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.77 GiB | ending 0.1924600452184677, calls 19, predictions [0.42425113916397095, 0.5527560710906982, 0.5510468482971191, 0.4338952302932739] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.77 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.77 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.77 GiB | ending 0.2500884532928467, calls 1, predictions [0.5136224031448364, 0.51397705078125, 0.5138164758682251, 0.5137636065483093] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.77 GiB | ending 0.19880273938179016, calls 36, predictions [0.45404180884361267, 0.5783681273460388, 0.5552966594696045, 0.46208494901657104] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.77 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | failed; peak 1.29 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | failed; peak 1.29 GiB | CLI exit 1; MSE [None, 0]; reconstruction [0, 0] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB |  |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB |  |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 3.98 GiB | exact 1.0; where 1.0; output 0.17501772940158844; reconstruction 0.00013363624748308212 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.82 GiB | exact 1.0; where 1.0; output 0.17500759661197662; reconstruction 0.0001300703443121165 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 3.98 GiB | exact 1.0; where 1.0; output 0.17498597502708435; reconstruction 0.00014014160842634737 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 3.73 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 3.99 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.13 GiB | exact 1.0; where 1.0; output 0.17621168494224548; reconstruction 0.0001343178446404636 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | passed; peak 4.17 GiB | exact 1.0; where 1.0; output 0.17650380730628967; reconstruction 3.343881689943373e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.17089150846004486; reconstruction 2.598520222818479e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | passed; peak 4.20 GiB | exact 1.0; where 1.0; output 0.1714356690645218; reconstruction 8.707165397936478e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1753479540348053; reconstruction 7.824605745554436e-06 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.18 GiB | exact 1.0; where 1.0; output 0.15691380202770233; reconstruction 0.00022510811686515808 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17066079378128052; reconstruction 3.220281359972432e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1372402012348175; reconstruction 0.00037931956467218697 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1748620867729187; reconstruction 4.929056103719631e-06 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | failed; peak 4.19 GiB | exact 0.5; where 1.0; output 0.16060544550418854; reconstruction 0.0001601710100658238 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17844851315021515; reconstruction 2.812110869854223e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17553657293319702; reconstruction 6.128316454123706e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17856711149215698; reconstruction 2.1216535969870165e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.1769602745771408; reconstruction 0.00011666114005493 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.17826008796691895; reconstruction 4.0347124013351277e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | passed; peak 4.19 GiB | exact 1.0; where 1.0; output 0.16989266872406006; reconstruction 4.4675194658339024e-05 |
