| Proof | Guarded result | Observations / diagnostic |
|---|---|---|
| test_grounded_xor.py::test_native_unseeded_xor[0-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[0-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[1-8] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-4] | passed; peak 0.43 GiB |  |
| test_grounded_xor.py::test_native_unseeded_xor[2-8] | passed; peak 0.43 GiB |  |
| test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0] | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1] | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2] | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent | passed; peak 1.06 GiB |  |
| test_concept_output.py::test_property_reverse_attributes_only_written_members | passed; peak 1.06 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs | passed; peak 0.66 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_convergence | passed; peak 0.66 GiB | ending 0.19469299912452698, calls 20, predictions [0.42779117822647095, 0.5447103381156921, 0.5487332344055176, 0.42992591857910156] |
| test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols | passed; peak 0.66 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces | passed; peak 0.66 GiB |  |
| test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal | passed; peak 0.66 GiB | ending 0.25002747774124146, calls 1, predictions [0.49914970993995667, 0.4990502595901489, 0.4989469051361084, 0.49895352125167847] |
| test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal | passed; peak 0.66 GiB | ending 0.19975103437900543, calls 183, predictions [0.4941566288471222, 0.6174347400665283, 0.49623385071754456, 0.3932897746562958] |
| test_mm_xor.py::TestMMXorConvergence::test_model_is_mental | passed; peak 0.66 GiB |  |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | passed; peak 1.09 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | passed; peak 1.10 GiB | CLI exit 0; MSE [0.0, 4]; reconstruction [4, 4] |
| test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | failed; peak 0.61 GiB | class accuracy [0.0] |
| test_basicmodel.py::TestSPNN::test_xor_training | passed; peak 0.41 GiB |  |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.17499390244483948; reconstruction 0.00013660475087817758 |
| test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.17499567568302155; reconstruction 0.00013989095168653876 |
| test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | passed; peak 4.52 GiB | exact 1.0; where 1.0; output 0.17499205470085144; reconstruction 0.0001360487804049626 |
| test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | passed; peak 4.23 GiB |  |
| test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | passed; peak 4.54 GiB |  |
| test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | passed; peak 4.51 GiB | exact 1.0; where 1.0; output 0.17469701170921326; reconstruction 0.0001373254635836929 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 1/15 | failed; peak 4.57 GiB | exact 0.5; where 1.0; output 0.157982736825943; reconstruction 0.00014625684707425535 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 2/15 | failed; peak 4.53 GiB | exact 0.0; where 1.0; output 0.1796693652868271; reconstruction 9.432889783056453e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 3/15 | failed; peak 4.52 GiB | exact 0.5; where 1.0; output 0.1788126677274704; reconstruction 9.824107110034674e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 4/15 | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.16727207601070404; reconstruction 0.0001917875197250396 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 5/15 | passed; peak 4.60 GiB | exact 1.0; where 1.0; output 0.17525909841060638; reconstruction 0.0003022501477971673 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 6/15 | passed; peak 4.55 GiB | exact 1.0; where 1.0; output 0.17428839206695557; reconstruction 0.00013321723963599652 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 7/15 | passed; peak 4.61 GiB | exact 1.0; where 1.0; output 0.17750048637390137; reconstruction 0.00011349684791639447 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 8/15 | failed; peak 4.45 GiB | exact 0.75; where 1.0; output 0.17337583005428314; reconstruction 0.00010296177788404748 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 9/15 | passed; peak 4.59 GiB | exact 1.0; where 1.0; output 0.17454840242862701; reconstruction 3.956450746045448e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 10/15 | passed; peak 4.50 GiB | exact 1.0; where 1.0; output 0.16164173185825348; reconstruction 0.0003231620939914137 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 11/15 | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.17597877979278564; reconstruction 8.607699419371784e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 12/15 | passed; peak 4.62 GiB | exact 1.0; where 1.0; output 0.1774345338344574; reconstruction 8.880436507752165e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 13/15 | passed; peak 4.54 GiB | exact 1.0; where 1.0; output 0.1661968231201172; reconstruction 0.0002812817692756653 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 14/15 | passed; peak 4.49 GiB | exact 1.0; where 1.0; output 0.1775154322385788; reconstruction 6.965352076804265e-05 |
| test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip — trial 15/15 | failed; peak 4.57 GiB | exact 0.0; where 0.3333333333333333; output 0.17724616825580597; reconstruction 0.0002565921167843044 |
